"""The Producer: one coordinator that makes a whole rough song.

It sequences the composer the creator already has - the tune engine, the
lyric writer, the singer, the beat, the arranger, the mixer - through the
seven production stages, and has the Critic review every stage before it
moves on.  Nothing here composes; the specialists do, and each is the
existing engine reached through the controller's own methods, so a song
the Producer makes is the same kind of song the creator can make by hand,
only walked end to end without them.

It is a state machine, not a script.  Every stage is a background job in
this application and every result comes back through ``pump`` on the
interface thread, so the Producer is *advanced* from ``pump`` rather than
running a loop of its own: it submits a stage, waits for the artifact to
land, describes it as evidence, hands the evidence to the Critic in a job
of its own, and acts on the verdict.  The window never waits on it.

A review is of a snapshot.  The Critic thinks for twenty seconds, and the
creator may edit the brief or write another tune while it does; so the
review runs against a copy of the journal, and its records are committed
only once the song's work ticket still holds.  A verdict about a song that
no longer exists is discarded, and the production stops rather than build
on it.

What the Critic cannot accept, the Producer may still pass - but only as a
decision recorded under its own name.  ``ProjectState.is_accepted`` stays
false for such a stage, and the report says which stages Codex accepted and
which the Producer proceeded past, so a song made with the reviewer
unavailable is never described as a reviewed one.

That is a policy, and it is explicit: ``on_blocked="proceed"`` (the
default) makes the song anyway and discloses every unreviewed stage, which
is what a working demo needs when the reviewer is down; ``on_blocked="stop"``
halts at the first stage the Critic could not review, which is what a
production that must be entirely reviewed needs.  A *revise* verdict is
neither: it is answered with another round until the budget is spent, and
only then does the policy apply - to exhaustion as to any other block.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..agent.knowledge import Lesson
from ..audio import dsp
from ..core.jobs import JobCancelled
from ..core.models import SectionKind
from ..music.structure import read_section_requests
from ..voice import mastering
from .contracts import (PRODUCTION_STAGES, Critic, InstrumentProfile,
                        ProjectState, Role, RoleAssignment, StageRecord,
                        Verdict, production_team, review_stage)
from .critic import CancelledReview
from .state import canonical_json

#: Below this a section of the vocal-only master is silent in fact, not
#: merely expected to be: a rest at -60 dB FS is thirty times quieter than
#: the quietest sung phrase measured on 2026-09-08 (-17.5 dB).
SILENCE_DB = -60.0

log = logging.getLogger("raaga.production")

#: The job targets a stage submits, so a cancel can reach them.
STAGE_TARGETS = {
    "tune": ("melody:all",),
    "lyrics": ("lyrics",),
    "voice": ("vocal",),
    "beat": ("beat:all", "render:beat"),
    "arrangement": ("arrangement:auto", "render:full"),
    "mix": ("render:full",),
}
REVIEW_TARGET = "production:review"

#: Which ProjectState field a stage's artifact reference lives in.
_STATE_REF = {"tune": "tune_ref", "lyrics": "lyric_ref", "beat": "beat",
              "arrangement": "arrangement", "mix": "mix"}

#: How many consecutive idle pumps before "nothing landed" is believed.  A
#: worker can finish between ``drain`` and ``advance`` inside one pump, with
#: its completion still queued; one more pump delivers it.
_IDLE_PUMPS_BEFORE_FAILURE = 3

#: Told to the Critic in every packet.  It cannot listen, and a reviewer
#: that blocks for want of ears blocks every stage; what only listening
#: could settle belongs under uncertainty.
REVIEW_SCOPE = (
    "The reviewer cannot hear audio and cannot open files.  Judge this stage "
    "on the facts in this packet, against the brief, as a guiding mentor: keep "
    "what works, name specific weaknesses with evidence, give actionable "
    "revisions with reasons, and reusable lessons.  What only listening could "
    "settle - timbre, diction, audible balance, felt emotion - goes under "
    "uncertainty, not into a block.  Block only when the packet is malformed, "
    "contradicts itself, or lacks the facts named in what_to_judge.")

WHAT_TO_JUDGE = {
    "brief": "Whether the brief gives the specialists enough to start: "
             "situation, emotional direction, language, raaga, tempo, "
             "duration, requested sections.",
    "tune": "The timed melody: raaga pitch inventory, section form, motif "
            "and its development, repetition, phrase pacing and rests, "
            "cadences.  The evaluator's scores are model assessments, "
            "supplied as evidence.",
    "lyrics": "Fit of syllables to notes per line, coverage of the sung "
              "sections, repetition and development.  When no language model "
              "wrote them the lines are transliterated syllables from the "
              "built-in lyric engine: judge fit and structure, not translation.",
    "voice": "The take's linkage to the tune and lyrics, coverage of the sung "
             "sections, level per section, technical measurements.  Timbre, "
             "diction and expression are listening matters.",
    "beat": "The timed strokes against the tala and tempo, density and "
            "variation across the sections, and whether the beat is in the "
            "arrangement.",
    "arrangement": "The cast, each track's timed regions against the sections, "
                   "roles and separation, and the vocal's place: it is mixed "
                   "from the vocal master take and is not a track.",
    "mix": "Section boundaries, level per section, loudness and peak, length "
           "against the timeline, linkage to the arrangement.  Balance and "
           "clarity are listening matters.",
}


def locate_codex(configured: str = "") -> str:
    """The Codex CLI executable the application should spawn, or "".

    A configured path wins.  Otherwise PATH, then the place the Codex
    desktop application keeps its bundled CLI on Windows, newest first -
    which is where it was found on Krish's machine, off PATH.
    """
    if configured and configured.strip():
        return configured.strip()
    found = shutil.which("codex")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates = glob.glob(os.path.join(local, "OpenAI", "Codex", "bin",
                                            "*", "codex.exe"))
        if candidates:
            return max(candidates, key=lambda p: os.path.getmtime(p))
    return ""


def _rms_db(audio: Optional[np.ndarray], sr: int, start: float, end: float) -> float:
    if audio is None or len(audio) == 0:
        return -120.0
    lo, hi = max(0, int(start * sr)), min(len(audio), int(end * sr))
    if hi <= lo:
        return -120.0
    return round(float(dsp.rms_db(audio[lo:hi])), 1)


def _digest(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:12]


class Producer:
    """Makes one whole song; see the module docstring."""

    def __init__(self, app, critic: Critic, *, max_rounds: int = 2,
                 seed: Optional[int] = None, on_blocked: str = "proceed",
                 journal_name: str = "production.json") -> None:
        if type(max_rounds) is not int or not 1 <= max_rounds <= 10:
            raise ValueError("Set a revision budget between 1 and 10")
        if on_blocked not in ("proceed", "stop"):
            raise ValueError("on_blocked is 'proceed' or 'stop'")
        self.app = app
        self.critic = critic
        self.max_rounds = max_rounds
        self.on_blocked = on_blocked
        self.seed = int(seed) if seed is not None else int(time.time()) % 9999
        self.journal_name = journal_name
        self.team: List[RoleAssignment] = production_team()
        self.state: Optional[ProjectState] = None
        self.journal_path: Optional[Path] = None
        self.phase = "idle"       # idle | working | landed | reviewing | done | failed | cancelled
        self.stage = PRODUCTION_STAGES[0]
        self.round = 1
        self.refs: Dict[str, str] = {}
        self.accepted = 0
        self.decisions = 0
        self.error = ""
        self.events: List[str] = []
        self.started_at = 0.0
        self.finished_at = 0.0
        self._generation: Optional[int] = None
        self._marker = 0
        self._idle_pumps = 0
        self._artifact: Any = None
        self._ref = ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.phase in ("done", "failed", "cancelled")

    @property
    def critic_available(self) -> bool:
        return bool(getattr(self.critic, "available", True))

    def critic_status(self) -> str:
        status = getattr(self.critic, "status", None)
        return status() if callable(status) else "reviewer configured"

    def start(self) -> bool:
        """Begin at the brief.  False, with a status, when it cannot."""
        app = self.app
        if self.phase != "idle":
            return False
        self.started_at = time.time()
        if not app.project.raaga.selected:
            suggestions = app.raaga_suggestions(limit=1)
            if not suggestions:
                self._fail("no raaga could be suggested for this brief")
                return False
            app.select_raaga(suggestions[0].name,
                             rationale="chosen by the Producer from the brief",
                             by_creator=False)
            self._note(f"raaga: {suggestions[0].name}, chosen from the brief")
        if app.project_dir is None:
            app.save()
        self.journal_path = Path(app.project_dir) / self.journal_name
        self._generation = app._project_generation
        brief = self._brief_dict()
        if self.journal_path.exists():
            # A song produced before keeps its journal: this run is a new
            # revision of it, with every earlier review still on record and
            # none of them counting for this one.
            try:
                self.state = ProjectState.load(self.journal_path)
            except Exception as exc:  # noqa: BLE001
                self._fail(f"the existing production journal could not be read "
                           f"({type(exc).__name__}: {exc}); nothing was overwritten")
                return False
            if self.state.project_id != app.project.project_id:
                self._fail("the production journal beside this song belongs to "
                           "another song; nothing was overwritten")
                return False
            self.state.invalidate_from("brief", "a new production was started")
            self.state.brief = brief
            self._note(f"continuing the journal at revision {self.state.revision}; "
                       f"{len(self.state.records)} earlier record(s) kept")
        else:
            self.state = ProjectState(project_id=app.project.project_id, brief=brief)
        self._note(f"production started: {app.project.title!r} in "
                   f"{app.project.raaga.selected}; critic: {self.critic_status()}; "
                   f"on a blocked review: {self.on_blocked}")
        app.status(f"Producing a whole song in {app.project.raaga.selected}...")
        self._begin_stage(PRODUCTION_STAGES[0])
        return True

    def cancel(self, reason: str = "cancelled by the creator") -> None:
        if self.finished:
            return
        for target in STAGE_TARGETS.get(self.stage, ()) + (REVIEW_TARGET,):
            self.app.jobs.cancel_target(target, reason="production cancelled")
        self.phase = "cancelled"
        self.error = reason
        self.finished_at = time.time()
        self._note(f"cancelled at {self.stage}: {reason}")
        self._save_journal()
        self.app.status(f"Production stopped at the {self.stage}: {reason}.")

    def advance(self) -> None:
        """Called from ``pump``: move the production on if its stage is ready."""
        if self.phase not in ("working", "landed"):
            return
        app = self.app
        if app._project_generation != self._generation:
            self._fail("the song changed while I was producing it")
            return
        if self.phase == "working":
            if app.jobs.active_jobs():
                self._idle_pumps = 0
                return
            if not self._landed():
                self._idle_pumps += 1
                if self._idle_pumps < _IDLE_PUMPS_BEFORE_FAILURE:
                    return
                self._fail(f"the {self.stage} did not arrive: {app.status_text}")
                return
            self.phase = "landed"
        try:
            self._artifact, self._ref = self._evidence()
        except Exception as exc:  # noqa: BLE001 - evidence must not crash the app
            log.exception("could not describe the %s", self.stage)
            self._fail(f"I could not describe the {self.stage} for review "
                       f"({type(exc).__name__})")
            return
        self.refs[self.stage] = self._ref
        self._record_ref(self.stage, self._ref)
        self._note(f"{self.stage}: {self._ref} landed (round {self.round})")
        self._review()

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def _begin_stage(self, stage: str) -> None:
        self.stage = stage
        self.round = 1
        self._submit()

    def _submit(self) -> None:
        """Ask the right specialist for this stage's artifact."""
        app, stage, project = self.app, self.stage, self.app.project
        seed = self.seed + 7919 * (self.round - 1)
        self._idle_pumps = 0
        if stage == "brief":
            self._marker = 0
            self.phase = "landed"
            return
        if stage == "tune":
            self._marker = len(project.melodies)
            app.generate_tune(seed=seed)
        elif stage == "lyrics":
            self._marker = len(project.lyrics)
            app.generate_lyrics(seed=seed)
        elif stage == "voice":
            self._marker = len(project.vocal_renders)
            app.render_vocal("master", autoplay=False)
        elif stage == "beat":
            self._marker = len(project.beats)
            app.generate_beat(seed=seed)
        elif stage == "arrangement":
            self._marker = len(project.arrangements)
            app.auto_arrange()
        elif stage == "mix":
            mix, arrangement = project.latest_mix("full"), project.arrangement()
            fresh = (mix is not None and arrangement is not None
                     and mix.arrangement_version == arrangement.version)
            if fresh and self.round == 1:
                # Arranging already mixed the song; that mix is the artifact.
                self._marker = len(project.mixes) - 1
            else:
                self._marker = len(project.mixes)
                app.render("full", autoplay=False)
        self.phase = "working"
        self._note(f"{stage}: round {self.round} submitted"
                   + (f" (seed {seed})" if stage in ("tune", "lyrics", "beat") else ""))

    def _landed(self) -> bool:
        project, stage = self.app.project, self.stage
        counts = {"brief": 1, "tune": len(project.melodies),
                  "lyrics": len(project.lyrics),
                  "voice": len(project.vocal_renders),
                  "beat": len(project.beats),
                  "arrangement": len(project.arrangements),
                  "mix": len(project.mixes)}
        return counts[stage] > self._marker

    def _record_ref(self, stage: str, ref: str) -> None:
        """The journal names the artifact under review, not only the Producer."""
        if self.state is None:
            return
        if stage == "voice":
            if ref not in self.state.vocal_takes:
                self.state.vocal_takes.append(ref)
        elif stage in _STATE_REF:
            setattr(self.state, _STATE_REF[stage], ref)

    def _next_stage(self) -> None:
        index = PRODUCTION_STAGES.index(self.stage) + 1
        if index >= len(PRODUCTION_STAGES):
            self._finish()
        else:
            self._begin_stage(PRODUCTION_STAGES[index])

    # ------------------------------------------------------------------
    # review
    # ------------------------------------------------------------------
    def _ticket(self) -> Dict[str, Any]:
        """What must still be true when the verdict lands."""
        app = self.app
        ticket = app.song_work_ticket()
        ticket["lyric_fingerprint"] = app.lyric_fingerprint(app.project.lyrics_version())
        ticket["voice_profile_id"] = app.current_voice().id
        ticket["brief_digest"] = _digest(self._brief_dict())
        ticket["artifact_ref"] = self._ref
        return ticket

    def _stale(self, ticket: Dict[str, Any]) -> str:
        app = self.app
        if app._project_generation != self._generation:
            return "the song changed while I was producing it"
        reason = app.stale_reason(ticket)
        if reason:
            return reason
        if ticket.get("brief_digest") != _digest(self._brief_dict()):
            return "the brief changed while I was producing"
        return ""

    def _review(self) -> None:
        stage, artifact, ref = self.stage, self._artifact, self._ref
        if not self.critic_available:
            self._blocked(f"the Codex reviewer is unavailable - {self.critic_status()}")
            return
        self.phase = "reviewing"
        round_number, max_rounds, critic = self.round, self.max_rounds, self.critic
        ticket = self._ticket()
        # The reviewer works on a copy.  Its records join the journal only
        # once the ticket has been checked on this thread.
        snapshot = ProjectState.from_dict(self.state.to_dict())
        already = len(snapshot.records)

        def work(ctx) -> Tuple[Verdict, List[StageRecord]]:  # noqa: ANN001
            ctx.progress(0.3, f"Codex reviews the {stage}")
            try:
                # ctx.cancelled is a property; the reviewer polls it every
                # 100 ms and stops the CLI child when it turns true.
                verdict = review_stage(critic, stage, snapshot, artifact,
                                       artifact_ref=ref, round_number=round_number,
                                       max_rounds=max_rounds,
                                       cancelled=lambda: ctx.cancelled)
            except CancelledReview as exc:
                # A cancellation is not a verdict: no record, no decision.
                raise JobCancelled(str(exc)) from exc
            return verdict, list(snapshot.records[already:])

        self.app.status(f"Codex is reviewing the {stage}...")
        self.app.jobs.submit("production.review", REVIEW_TARGET, work,
                             on_done=lambda result: self._reviewed(result, ticket),
                             on_error=self._review_crashed,
                             on_cancelled=self._review_cancelled,
                             description=f"Codex reviews the {stage}",
                             provider="codex")

    def _reviewed(self, result: Tuple[Verdict, List[StageRecord]],
                  ticket: Dict[str, Any]) -> None:
        if self.phase != "reviewing":
            return
        verdict, records = result
        stale = self._stale(ticket)
        if stale:
            # The verdict is about a song that no longer exists.  Nothing of
            # it is committed, and nothing is built on it.
            self._note(f"{self.stage}: a verdict arrived for a song that changed "
                       f"({stale}); discarded")
            self._fail(f"the song changed during the {self.stage} review: {stale}")
            return
        for record in records:
            self.state.add_record(record)
        if not self._save_journal():
            return
        stage = self.stage
        if verdict.accept:
            self.accepted += 1
            self._note(f"{stage}: Codex accepted round {self.round} - {verdict.reason}")
            self._next_stage()
            return
        if verdict.blocked:
            self._blocked(f"Codex could not review this stage ({verdict.reason})")
            return
        advice = "; ".join(verdict.revisions)
        self._note(f"{stage}: Codex asks for a revision after round {self.round} - {advice}")
        if self.round >= self.max_rounds:
            exhausted = review_stage(self.critic, stage, self.state, self._artifact,
                                     artifact_ref=self._ref,
                                     round_number=self.round + 1,
                                     max_rounds=self.max_rounds)
            self._blocked(f"the revision budget of {self.max_rounds} is spent "
                          f"({exhausted.reason}); round {self.round} stands with "
                          f"the Critic's advice on record: {advice}")
            return
        self._apply_advice(verdict)
        self.round += 1
        self._submit()

    def _review_cancelled(self) -> None:
        """The review job was cancelled from outside the Producer - by a
        superseding job or a shutdown.  Nothing was recorded; the
        production cannot continue without its verdict, and says so."""
        if self.phase != "reviewing":
            return
        self._fail(f"the {self.stage} review was cancelled before it answered")

    def _review_crashed(self, exc: BaseException) -> None:
        if self.phase != "reviewing":
            return
        # review_stage already turns its own failures into a blocked verdict;
        # this is the job machinery failing around it.
        self._blocked(f"the review job failed ({type(exc).__name__})")

    def _blocked(self, why: str) -> None:
        """A stage the Critic could not review, or would not accept within
        the budget: the policy decides, and the same policy every time."""
        if self.on_blocked == "stop":
            self._fail(f"the Critic could not accept the {self.stage}: {why}")
            return
        self._decide(f"{why}; the Producer proceeded without an accepted review")

    def _apply_advice(self, verdict: Verdict) -> None:
        """Turn the Critic's revisions into the next attempt's guidance.

        The tune engine learns from lessons: a lesson's kind becomes a
        lever on the next composition.  The Critic's advice enters that way,
        under the Critic's own name and at a reviewer's confidence - never as
        the creator's testimony, which ``give_feedback`` would make it, with
        the phrase-confidence changes and high-weight lessons that carries.
        The other specialists are reseeded; the advice stays on record.
        """
        advice = "; ".join(verdict.revisions)
        if not advice:
            return
        app = self.app
        app.project.log_history("critic.advice", advice[:200])
        if self.stage != "tune":
            return
        try:
            kinds = list(app.agent.feedback_kinds(advice)) or ["critic_advice"]
            raaga = app.project.raaga.selected
            for kind in kinds:
                app.agent.repo.add_lesson(Lesson(
                    raaga=raaga, kind=kind, dimension="critic", task="composition",
                    method="codex critic", failure_reason=advice[:200],
                    correction=advice[:200], confidence=0.6,
                    source_run=f"production:{app.project.project_id}:"
                               f"r{self.state.revision if self.state else 0}"))
            self._note(f"{self.stage}: advice filed as lesson(s) {', '.join(kinds)} "
                       f"under the Critic's name")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not file the Critic's advice as a lesson: %s", exc)

    def _decide(self, reason: str) -> None:
        """Proceed past a stage the Critic did not accept, under the
        Producer's own name.  Never counted as an acceptance."""
        record = StageRecord(stage=self.stage, role="producer",
                             artifact_ref=self._ref, rationale=reason,
                             verdict="accept", round=self.round,
                             revision=self.state.revision, provider="producer")
        self.state.add_record(record)
        self.decisions += 1
        self._note(f"{self.stage}: Producer decision - {reason}")
        if not self._save_journal():
            return
        self._next_stage()

    # ------------------------------------------------------------------
    # ending
    # ------------------------------------------------------------------
    def _finish(self) -> None:
        self.phase = "done"
        self.finished_at = time.time()
        app = self.app
        mix = app.project.latest_mix("full")
        self._note(f"done: Codex accepted {self.accepted} of {len(PRODUCTION_STAGES)} "
                   f"stages, Producer decided {self.decisions}"
                   + (f"; mix {mix.duration:.1f}s at {mix.audio_path}" if mix else ""))
        if not self._save_journal():
            return
        self._write_report()
        app.project.log_history("production.done", self.summary())
        app.dirty = True
        try:
            app.save()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save the produced song: %s", exc)
        app.status(f"Whole song produced - {self.summary()}")
        if app.on_project_changed:
            app.on_project_changed()

    def _fail(self, reason: str) -> None:
        self.phase = "failed"
        self.error = reason
        self.finished_at = time.time()
        self._note(f"failed at {self.stage}: {reason}")
        self._save_journal()
        self.app.status(f"Production stopped at the {self.stage}: {reason}.")

    def _save_journal(self) -> bool:
        """Write the journal.  A journal that cannot be written ends the
        production visibly: a record nobody can read is not a record."""
        if self.state is None or self.journal_path is None:
            return True
        try:
            self.state.save(self.journal_path)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("could not write the production journal: %s", exc)
            self._note(f"journal not written: {type(exc).__name__}: {exc}")
            if self.phase not in ("failed", "cancelled"):
                self.phase = "failed"
                self.error = f"the production journal could not be written: {exc}"
                self.finished_at = time.time()
                self.app.status(f"Production stopped at the {self.stage}: "
                                f"{self.error}")
            return False

    def _write_report(self) -> None:
        if self.journal_path is None:
            return
        try:
            path = self.journal_path.with_name("production_report.txt")
            path.write_text(self.report(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write the production report: %s", exc)

    def _note(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"{stamp}  {text}")
        log.info("production: %s", text)

    # ------------------------------------------------------------------
    # evidence: what each stage hands the Critic
    # ------------------------------------------------------------------
    def _brief_dict(self) -> Dict[str, Any]:
        brief, raaga = self.app.project.brief, self.app.project.raaga
        wants = read_section_requests(brief.notes, brief.feel, brief.situation)
        return {
            "title": self.app.project.title, "situation": brief.situation,
            "mood": brief.mood, "feel": brief.feel, "language": brief.language,
            "song_type": brief.song_type,
            "duration_target": float(brief.duration_target),
            "tempo_preference": brief.tempo_preference, "tala": brief.tala,
            "vocal_feel": brief.vocal_feel, "notes": brief.notes,
            "instruments_preferred": list(brief.instruments_preferred),
            "instruments_avoided": list(brief.instruments_avoided),
            "raaga": {"selected": raaga.selected, "rationale": raaga.rationale,
                      "alternatives": list(raaga.alternatives)},
            "requested_sections": [k.value for k in wants.wanted],
            "refused_sections": [k.value for k in wants.refused],
        }

    def _timeline(self) -> Dict[str, Any]:
        """One authoritative timeline, in every packet after the tune."""
        app = self.app
        melody = app.project.melody()
        if melody is None:
            return {}
        notes_end = max((n.end for n in melody.notes), default=0.0)
        tala = app.current_tala()
        return {
            "authority": "the tune's sections define the song; the singer "
                         "renders the song length plus a one-second tail and "
                         "the mixer adds a half-second tail, so renders run "
                         "longer than the last section ends",
            "sections": [{"name": s.name, "kind": s.kind.value,
                          "start": round(s.start, 2), "end": round(s.end, 2)}
                         for s in melody.sections],
            "sections_end": round(max((s.end for s in melody.sections), default=0.0), 2),
            "last_note_end": round(notes_end, 2),
            "tune_duration": round(melody.duration, 2),
            "tempo_bpm": melody.tempo_bpm, "beats_per_cycle": melody.beats_per_cycle,
            "tala": getattr(tala, "name", ""),
        }

    def _packet(self, stage: str, body: Dict[str, Any]) -> Dict[str, Any]:
        packet = {"stage": stage, "review_scope": REVIEW_SCOPE,
                  "what_to_judge": WHAT_TO_JUDGE[stage]}
        if stage != "brief":
            packet["timeline"] = self._timeline()
        packet.update(body)
        return packet

    def _evidence(self) -> Tuple[Dict[str, Any], str]:
        return getattr(self, f"_evidence_{self.stage}")()

    def _evidence_brief(self) -> Tuple[Dict[str, Any], str]:
        body = {"by": "producer", **self._brief_dict()}
        return self._packet("brief", body), f"brief:{_digest(body)}"

    def _evidence_tune(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        melody = app.project.melody()
        raaga = app.composing_raaga()
        evaluation = app.agent.evaluator(raaga.name).evaluate(
            melody.notes, raaga, tonic_midi=melody.tonic_midi,
            brief=app.project.brief, tempo_bpm=melody.tempo_bpm,
            expected_seconds=app.project.brief.duration_target,
            learned_phrases=app.agent.phrase_bank(raaga.name))
        names = {s.id: s.name for s in melody.sections}
        timed = []
        previous_end = 0.0
        for note in melody.notes:
            rest = round(note.start - previous_end, 2)
            entry = {"t": round(note.start, 2), "d": round(note.duration, 2),
                     "swara": note.swara, "midi": note.midi,
                     "section": names.get(note.section_id, "")}
            if note.gamaka:
                entry["gamaka"] = note.gamaka
            if rest > 0.05:
                entry["rest_before"] = rest
            timed.append(entry)
            previous_end = note.end
        sections = []
        for section in melody.sections:
            notes = [n for n in melody.notes if n.section_id == section.id]
            sections.append({"name": section.name, "kind": section.kind.value,
                             "start": round(section.start, 2),
                             "end": round(section.end, 2), "notes": len(notes),
                             "swaras": " ".join(n.swara for n in notes)})
        body = {
            "by": "producer (melody engine)",
            "raaga": melody.raaga, "tonic_midi": melody.tonic_midi,
            "tempo_bpm": melody.tempo_bpm, "beats_per_cycle": melody.beats_per_cycle,
            "duration": round(melody.duration, 2), "note_count": len(melody.notes),
            "sections": sections,
            "notes": timed,
            "evaluation": {
                "overall": round(evaluation.overall(), 3),
                "scores": {k: round(v, 3) for k, v in evaluation.scores.items()},
                "strengths": list(evaluation.strengths)[:8],
                "mistakes": list(evaluation.mistakes)[:8],
                "recommendation": evaluation.recommendation,
                "notes_examined": evaluation.notes_examined,
            },
            "validation": list(melody.validation)[:12],
            "plan_notes": list(melody.plan_notes),
            "guidance": melody.guidance_note,
        }
        return self._packet("tune", body), f"tune:v{melody.version}"

    def _evidence_lyrics(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        lyrics, melody = app.project.lyrics_version(), app.project.melody()
        names = {s.id: s.name for s in melody.sections} if melody else {}
        lines = []
        for line in lyrics.lines[:48]:
            lines.append({"section": names.get(line.section_id, line.section_id),
                          "text": line.text, "syllables": list(line.syllables),
                          "syllable_count": len(line.syllables),
                          "note_indices": list(line.note_indices),
                          "note_count": len(line.note_indices),
                          "start": round(line.start, 2), "end": round(line.end, 2)})
        written_by = ("a language model" if getattr(app.providers, "llm", None)
                      and getattr(app.providers.llm, "available", False)
                      else "the built-in lyric engine (transliterated syllables; "
                           "no translation exists)")
        body = {"by": "lyrics", "language": lyrics.language, "written_by": written_by,
                "for_tune": f"tune:v{lyrics.melody_version}",
                "line_count": len(lyrics.lines), "lines": lines,
                "lines_shown": len(lines),
                "lines_omitted": max(0, len(lyrics.lines) - len(lines)),
                "note": "a syllable may span several notes (melisma); note_indices "
                        "lists every note the line sings, in order",
                "alignment": app.lyric_alignment()}
        return self._packet("lyrics", body), f"lyrics:v{lyrics.version}"

    def _evidence_voice(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        take = app.project.vocal_master or app.project.latest_vocal
        melody = app.project.melody()
        rendered = app.rendered("vocal_master")
        audio = rendered.audio if rendered is not None else None
        sr = app.sample_rate
        sections = melody.sections if melody else []
        sung = [s for s in sections if not s.kind.instrumental]
        instrumental = [s for s in sections if s.kind.instrumental]
        # Expected and measured are two different facts, and the Critic gets
        # both: the singer is *meant* to rest through the instrumental
        # sections, and here is what the take actually contains there.
        measured = []
        for s in instrumental:
            level = _rms_db(audio, sr, s.start, s.end)
            measured.append({"name": s.name, "start": round(s.start, 2),
                             "end": round(s.end, 2), "rms_db": level,
                             "measured_silent": bool(audio is not None and level < SILENCE_DB)})
        body = {"by": "singer-1", "voice": app.current_voice().name, "take": take.id,
                "kind": take.kind, "duration": round(take.duration, 2),
                "for_tune": f"tune:v{take.melody_version}",
                "for_lyrics": f"lyrics:v{take.lyrics_version}",
                "sung_sections": [{"name": s.name, "start": round(s.start, 2),
                                   "end": round(s.end, 2),
                                   "rms_db": _rms_db(audio, sr, s.start, s.end)}
                                  for s in sung],
                "expected_silent_sections": [s.name for s in instrumental],
                "instrumental_sections": measured,
                "silence_threshold_db": SILENCE_DB,
                "measurement": "RMS in dB FS over the vocal-only master, per section"
                               + ("" if audio is not None else
                                  "; no audio was available to measure"),
                "report": mastering.report(audio, sr) if audio is not None else ""}
        return self._packet("voice", body), f"voice:{take.id}"

    def _evidence_beat(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        beat = app.project.beat()
        arrangement = app.project.arrangement()
        in_song = bool(arrangement and any(
            r.meta.get("beat_version") == str(beat.version)
            for t in arrangement.tracks for r in t.regions))
        tala = app.current_tala()
        aksharas = int(getattr(tala, "aksharas", 8) or 8)
        cycle_seconds = aksharas * 60.0 / max(1, beat.tempo_bpm)
        # Two cycles written out; the rest as counts per section.  The full
        # stroke list is what pushed a review past its time limit, and a
        # steady pattern says everything in its first two cycles.
        first_two = [{"t": round(n.start, 3), "d": round(n.duration, 3),
                      "stroke": n.swara, "velocity": n.velocity}
                     for n in beat.notes if n.start < 2 * cycle_seconds]
        melody = app.project.melody()
        per_section = []
        for section in (melody.sections if melody else []):
            inside = [n for n in beat.notes if section.start <= n.start < section.end]
            per_section.append({"name": section.name, "strokes": len(inside),
                                "strokes_per_cycle": round(
                                    len(inside) / max(0.01, (section.end - section.start)
                                                      / cycle_seconds), 1)})
        body = {"by": "percussion-1", "tala": beat.tala, "aksharas_per_cycle": aksharas,
                "cycle_seconds": round(cycle_seconds, 2),
                "tempo_bpm": beat.tempo_bpm, "density": beat.density,
                "duration": round(beat.duration, 2), "stroke_count": len(beat.notes),
                "first_two_cycles": first_two, "per_section": per_section,
                "summary": beat.summary(), "in_the_arrangement": in_song}
        return self._packet("beat", body), f"beat:v{beat.version}"

    def _evidence_arrangement(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        arrangement = app.project.arrangement()
        melody = app.project.melody()
        lead = app.cast_lead()
        self._bind_instruments(arrangement)
        assigned = {a.profile.description: a.id for a in self.team
                    if a.profile is not None}
        sections = melody.sections if melody else []
        tracks = []
        for track in arrangement.tracks:
            regions = []
            for region in track.regions:
                covers = [s.name for s in sections
                          if region.start < s.end and region.end > s.start]
                regions.append({"start": round(region.start, 2),
                                "end": round(region.end, 2), "role": region.role,
                                "notes": len(region.notes), "covers": covers,
                                "generated_by": region.generated_by})
            tracks.append({"label": track.label, "instrument": track.instrument,
                           "role": track.role, "gain": round(track.gain, 2),
                           "pan": round(track.pan, 2), "mute": track.mute,
                           "played_by": assigned.get(track.id, ""),
                           "regions": regions})
        body = {"by": "producer (casting)", "lead": lead.describe(),
                "vocal": "mixed from the vocal master take; not a track here",
                "tracks": tracks,
                "team": [{"id": a.id, "role": a.role.value,
                          "instrument": a.profile.instrument if a.profile else "",
                          "has_track": bool(a.profile and a.profile.description in
                                            {t.id for t in arrangement.tracks})}
                         for a in self.team]}
        return self._packet("arrangement", body), f"arrangement:v{arrangement.version}"

    def _evidence_mix(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        mix = app.project.latest_mix("full")
        melody = app.project.melody()
        rendered = app.rendered("full")
        audio = rendered.audio if rendered is not None else None
        sr = app.sample_rate
        settings = app.project.mix_settings
        body = {"by": "producer", "version": mix.version,
                "duration": round(mix.duration, 2),
                "loudness_db": round(mix.loudness_db, 1),
                "loudness_method": "K-weighted, whole file, dB",
                "peak_dbfs": (round(float(dsp.peak_db(audio)), 1)
                              if audio is not None else None),
                "true_peak": "not measured",
                "path": mix.audio_path,
                "for_arrangement": f"arrangement:v{mix.arrangement_version}",
                "for_vocal": f"voice:{app.project.vocal_master.id}"
                             if app.project.vocal_master else "",
                "sections": [{"name": s.name, "start": round(s.start, 2),
                              "end": round(s.end, 2),
                              "rms_db": _rms_db(audio, sr, s.start, s.end)}
                             for s in (melody.sections if melody else [])],
                "measurement": "RMS in dB FS over the full mix, per section",
                "settings": {"vocal_gain": round(settings.vocal_gain, 2)}}
        return self._packet("mix", body), f"mix:v{mix.version}"

    def _bind_instruments(self, arrangement) -> None:
        """Give the team's instrumental roles the instruments actually cast.

        Instrument identity is a profile assigned as the song is made, not a
        class per instrument: the four melody roles take the arrangement's
        melodic tracks in order, the percussion roles its rhythm tracks.
        """
        melodic, rhythm, drone = [], [], []
        for track in arrangement.tracks:
            role = (track.role or "").lower()
            if any(r.meta.get("beat_version") for r in track.regions) or \
                    any(w in role for w in ("beat", "percussion", "rhythm")):
                rhythm.append(track)
            elif "drone" in role or "tanpura" in track.instrument.lower():
                drone.append(track)
            else:
                melodic.append(track)
        pools = {Role.MELODY: melodic, Role.PERCUSSION: rhythm, Role.DRONE: drone}
        bound: List[RoleAssignment] = []
        for member in self.team:
            if member.profile is None:
                bound.append(member)
                continue
            pool = pools.get(member.role, [])
            if pool:
                track = pool.pop(0)
                profile = InstrumentProfile(member.profile.key, track.instrument,
                                            member.role, track.id)
                bound.append(RoleAssignment(member.id, member.role, profile))
            else:
                bound.append(member)
        self.team = bound

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def summary(self) -> str:
        total = len(PRODUCTION_STAGES)
        parts = [f"Codex accepted {self.accepted} of {total} stages"]
        if self.decisions:
            parts.append(f"the Producer decided {self.decisions}")
        if self.phase != "done":
            parts.append(f"{self.phase} at the {self.stage}"
                         + (f": {self.error}" if self.error else ""))
        return "; ".join(parts)

    def report(self) -> str:
        app = self.app
        lines = [f"Whole song production - {app.project.title!r} in "
                 f"{app.project.raaga.selected or 'no raaga'}",
                 f"State: {self.phase}.  {self.summary()}.",
                 f"Critic: {self.critic_status()}",
                 f"On a blocked review: {self.on_blocked}", ""]
        if self.state is not None:
            lines.append(f"Journal revision {self.state.revision}; "
                         f"{len(self.state.records)} record(s)")
            lines.append("Stage        Artifact               Verdict   By        Round  Reason")
            for record in self.state.records:
                if record.revision != self.state.revision:
                    continue
                who = record.provider or record.role
                lines.append(f"{record.stage:<12} {record.artifact_ref:<22} "
                             f"{record.verdict:<9} {who:<9} {record.round:<6} "
                             f"{record.rationale}")
                for item in record.revisions:
                    lines.append(f"{'':<12} revision: {item}")
                for item in record.lessons:
                    lines.append(f"{'':<12} lesson: {item}")
            lines.append("")
        mix = app.project.latest_mix("full")
        if mix:
            lines.append(f"Output: {mix.audio_path} ({mix.duration:.1f} s, "
                         f"{mix.loudness_db:.1f} dB)")
        take = app.project.vocal_master
        if take and take.audio_path:
            lines.append(f"Vocal master: {take.audio_path}")
        if self.journal_path:
            lines.append(f"Journal: {self.journal_path}")
        lines.append("")
        lines.append("Team:")
        for member in self.team:
            instrument = f" - {member.profile.instrument}" if member.profile else ""
            lines.append(f"  {member.id:<14} {member.role.value}{instrument}")
        lines.append("")
        lines.append("Log:")
        lines.extend(f"  {event}" for event in self.events)
        return "\n".join(lines)
