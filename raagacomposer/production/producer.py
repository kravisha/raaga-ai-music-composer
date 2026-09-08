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
only then does the policy apply.
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

from ..audio import dsp
from ..core.models import SectionKind
from ..music.structure import read_section_requests
from ..voice import mastering
from .contracts import (PRODUCTION_STAGES, Critic, InstrumentProfile,
                        ProjectState, Role, RoleAssignment, StageRecord,
                        Verdict, production_team, review_stage)
from .state import canonical_json

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

#: How many consecutive idle pumps before "nothing landed" is believed.  A
#: worker can finish between ``drain`` and ``advance`` inside one pump, with
#: its completion still queued; one more pump delivers it.
_IDLE_PUMPS_BEFORE_FAILURE = 3


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


def _rms_db(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    if audio is None or len(audio) == 0:
        return -120.0
    lo, hi = max(0, int(start * sr)), min(len(audio), int(end * sr))
    if hi <= lo:
        return -120.0
    return round(float(dsp.rms_db(audio[lo:hi])), 1)


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
        self.state = ProjectState(project_id=app.project.project_id,
                                  brief=self._brief_dict())
        self._note(f"production started: {app.project.title!r} in "
                   f"{app.project.raaga.selected}; critic: {self.critic_status()}")
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

    def _next_stage(self) -> None:
        index = PRODUCTION_STAGES.index(self.stage) + 1
        if index >= len(PRODUCTION_STAGES):
            self._finish()
        else:
            self._begin_stage(PRODUCTION_STAGES[index])

    # ------------------------------------------------------------------
    # review
    # ------------------------------------------------------------------
    def _review(self) -> None:
        stage, state, artifact, ref = self.stage, self.state, self._artifact, self._ref
        if not self.critic_available:
            self._blocked(f"the Codex reviewer is unavailable - {self.critic_status()}")
            return
        self.phase = "reviewing"
        round_number, max_rounds, critic = self.round, self.max_rounds, self.critic

        def work(ctx) -> Verdict:  # noqa: ANN001
            ctx.progress(0.3, f"Codex reviews the {stage}")
            return review_stage(critic, stage, state, artifact, artifact_ref=ref,
                                round_number=round_number, max_rounds=max_rounds)

        self.app.status(f"Codex is reviewing the {stage}...")
        self.app.jobs.submit("production.review", REVIEW_TARGET, work,
                             on_done=self._reviewed,
                             on_error=self._review_crashed,
                             description=f"Codex reviews the {stage}",
                             provider="codex")

    def _reviewed(self, verdict: Verdict) -> None:
        if self.phase != "reviewing":
            return
        self._save_journal()
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
            self._decide(f"the revision budget of {self.max_rounds} is spent "
                         f"({exhausted.reason}); the Producer kept round "
                         f"{self.round} with the Critic's advice on record: {advice}")
            return
        self._apply_advice(verdict)
        self.round += 1
        self._submit()

    def _review_crashed(self, exc: BaseException) -> None:
        if self.phase != "reviewing":
            return
        # review_stage already turns its own failures into a blocked verdict;
        # this is the job machinery failing around it.
        self._blocked(f"the review job failed ({type(exc).__name__})")

    def _blocked(self, why: str) -> None:
        """A stage the Critic could not review: the policy decides."""
        if self.on_blocked == "stop":
            self._fail(f"the Critic could not review the {self.stage}: {why}")
            return
        self._decide(f"{why}; the Producer proceeded without a review")

    def _apply_advice(self, verdict: Verdict) -> None:
        """Turn the Critic's revisions into the next attempt's guidance.

        The tune engine already learns from feedback - a creator's words
        become lessons that guide the next composition - so the Critic's
        advice enters by the same door.  The other specialists are reseeded;
        their advice stays on record for the creator and the learner.
        """
        advice = "; ".join(verdict.revisions)
        if self.stage == "tune" and advice:
            try:
                self.app.give_feedback(f"The Critic asks: {advice}")
            except Exception as exc:  # noqa: BLE001
                log.warning("could not pass the Critic's advice to the agent: %s", exc)

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
        self._save_journal()
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
        self._save_journal()
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

    def _save_journal(self) -> None:
        if self.state is None or self.journal_path is None:
            return
        try:
            self.state.save(self.journal_path)
        except Exception as exc:  # noqa: BLE001 - the journal is a record, not the song
            log.error("could not write the production journal: %s", exc)
            self._note(f"journal not written: {type(exc).__name__}: {exc}")

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

    def _evidence(self) -> Tuple[Dict[str, Any], str]:
        return getattr(self, f"_evidence_{self.stage}")()

    def _evidence_brief(self) -> Tuple[Dict[str, Any], str]:
        data = {"stage": "brief", "by": "producer", **self._brief_dict()}
        digest = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:12]
        return data, f"brief:{digest}"

    def _evidence_tune(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        melody = app.project.melody()
        raaga = app.composing_raaga()
        evaluation = app.agent.evaluator(raaga.name).evaluate(
            melody.notes, raaga, tonic_midi=melody.tonic_midi,
            brief=app.project.brief, tempo_bpm=melody.tempo_bpm,
            expected_seconds=app.project.brief.duration_target,
            learned_phrases=app.agent.phrase_bank(raaga.name))
        sections = []
        for section in melody.sections:
            notes = [n for n in melody.notes if n.section_id == section.id]
            sections.append({"name": section.name, "kind": section.kind.value,
                             "start": round(section.start, 2),
                             "end": round(section.end, 2), "notes": len(notes),
                             "swaras": " ".join(n.swara for n in notes[:24])})
        data = {
            "stage": "tune", "by": "producer (melody engine)",
            "raaga": melody.raaga, "tempo_bpm": melody.tempo_bpm,
            "beats_per_cycle": melody.beats_per_cycle,
            "duration": round(melody.duration, 2), "note_count": len(melody.notes),
            "sections": sections,
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
        return data, f"tune:v{melody.version}"

    def _evidence_lyrics(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        lyrics, melody = app.project.lyrics_version(), app.project.melody()
        names = {s.id: s.name for s in melody.sections} if melody else {}
        lines = [{"section": names.get(line.section_id, line.section_id),
                  "text": line.text, "syllables": len(line.syllables),
                  "notes": len(line.note_indices)} for line in lyrics.lines[:40]]
        data = {"stage": "lyrics", "by": "lyrics", "language": lyrics.language,
                "for_tune": f"tune:v{lyrics.melody_version}",
                "line_count": len(lyrics.lines), "lines": lines,
                "alignment": app.lyric_alignment()}
        return data, f"lyrics:v{lyrics.version}"

    def _evidence_voice(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        take = app.project.vocal_master or app.project.latest_vocal
        melody = app.project.melody()
        rendered = app.rendered("vocal_master")
        audio = rendered.audio if rendered is not None else None
        sr = app.sample_rate
        sung = [s for s in melody.sections if not s.kind.instrumental] if melody else []
        data = {"stage": "voice", "by": "singer-1",
                "voice": app.current_voice().name, "take": take.id,
                "kind": take.kind, "duration": round(take.duration, 2),
                "for_tune": f"tune:v{take.melody_version}",
                "for_lyrics": f"lyrics:v{take.lyrics_version}",
                "sections": [{"name": s.name,
                              "rms_db": _rms_db(audio, sr, s.start, s.end)}
                             for s in sung],
                "report": mastering.report(audio, sr) if audio is not None else ""}
        return data, f"voice:{take.id}"

    def _evidence_beat(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        beat = app.project.beat()
        arrangement = app.project.arrangement()
        in_song = bool(arrangement and any(
            r.meta.get("beat_version") == str(beat.version)
            for t in arrangement.tracks for r in t.regions))
        data = {"stage": "beat", "by": "percussion-1", "tala": beat.tala,
                "tempo_bpm": beat.tempo_bpm, "density": beat.density,
                "duration": round(beat.duration, 2), "note_count": len(beat.notes),
                "summary": beat.summary(), "in_the_song": in_song}
        return data, f"beat:v{beat.version}"

    def _evidence_arrangement(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        arrangement = app.project.arrangement()
        lead = app.cast_lead()
        self._bind_instruments(arrangement)
        assigned = {a.profile.description: a.id for a in self.team
                    if a.profile is not None}
        tracks = [{"label": t.label, "instrument": t.instrument, "role": t.role,
                   "regions": len(t.regions), "gain": round(t.gain, 2),
                   "pan": round(t.pan, 2),
                   "played_by": assigned.get(t.id, "")} for t in arrangement.tracks]
        data = {"stage": "arrangement", "by": "producer (casting)",
                "lead": lead.describe(), "tracks": tracks,
                "team": [{"id": a.id, "role": a.role.value,
                          "instrument": a.profile.instrument if a.profile else ""}
                         for a in self.team]}
        return data, f"arrangement:v{arrangement.version}"

    def _evidence_mix(self) -> Tuple[Dict[str, Any], str]:
        app = self.app
        mix = app.project.latest_mix("full")
        melody = app.project.melody()
        rendered = app.rendered("full")
        audio = rendered.audio if rendered is not None else None
        sr = app.sample_rate
        settings = app.project.mix_settings
        data = {"stage": "mix", "by": "producer", "version": mix.version,
                "duration": round(mix.duration, 2),
                "loudness_db": round(mix.loudness_db, 1), "path": mix.audio_path,
                "for_arrangement": f"arrangement:v{mix.arrangement_version}",
                "sections": [{"name": s.name,
                              "rms_db": _rms_db(audio, sr, s.start, s.end)}
                             for s in (melody.sections if melody else [])],
                "settings": {"vocal_gain": round(settings.vocal_gain, 2)}}
        return data, f"mix:v{mix.version}"

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
                 f"Critic: {self.critic_status()}", ""]
        if self.state is not None:
            lines.append("Stage        Artifact               Verdict   By        Round  Reason")
            for record in self.state.records:
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
