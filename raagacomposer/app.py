"""Application controller (spec section 12.2).

Everything the UI does goes through here: project lifecycle, the creative
workflow, background jobs, playback, voice commands, undo and export.  The UI
holds no musical logic; it renders this object's state and calls its methods.

Threading rule: worker functions compute and return, they never mutate the
project.  Completion callbacks run on the UI thread via
:meth:`JobManager.drain`, and only they write to project state.
"""
from __future__ import annotations

import copy
import hashlib
import queue
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agent.guidance import build_guidance
from .agent.knowledge import Lesson
from .agent.music_agent import MusicAgent
from .training.controller import TrainingController
from .kb.context import KnowledgeContextBuilder
from .kb.service import KnowledgeBaseService
from .audio import export as export_engine
from .audio.playback import PlaybackEngine
from .core import provenance
from .core.actions import ActionState, ActionStatus
from .core.jobs import JobCancelled, JobContext, JobManager
from .core.logging_setup import export_diagnostics, get_logger, setup_logging
from .core.models import (ApprovalState, ArrangementVersion, BeatVersion,
                          CreativeBrief,
                          ErrorRecord, JobRecord, LyricsVersion, MelodyVersion,
                          MixVersion, Project, Section, Stage, VocalDirection,
                          VocalRender, VoiceProfile)
from .core.persistence import ProjectStore
from .core.settings import Settings
from .core.versioning import (LockedContentError, UndoManager,
                              assert_melody_editable)
from .lyrics import fitting as lyric_fitting
from .lyrics import generator as lyric_generator
from .music import arrangement as arranger
from .music import instruments as catalog
from .music import melody as melody_engine
from .music import mixer
from .music.melody import MelodyOptions
from .music.structure import plan_sections, read_section_requests
from .music.synth import render_notes
from .music.validator import validate
from .providers import registry as provider_registry
from .providers.status import ProviderStatus
from .providers.status import provider_statuses as _provider_statuses
from .raaga import audition
from .raaga import vocabulary
from .raaga.library import Raaga, library as raaga_library
from .raaga.selection import (RaagaSuggestion, expand_feel_words, infer_tempo,
                              suggest as suggest_raagas)
from .speech.capture import CaptureState, VoiceInputManager
from .speech.context import ConversationContext
from .speech.intent import (Command, describe, interpret,
                            unavailable_instrument)
from .speech.timeline_parser import TimeSpec
from .voice import mastering
from .voice.profiles import VoiceProfileManager

log = get_logger("app")

RENDER_KINDS = ("tune", "vocal_preview", "vocal_master", "instrumental", "full")


def _normalize_suggestion(s) -> None:
    """Give any raaga suggestion object a ``reason`` and a ``confidence`` in
    [0, 1], whatever it started with (v0.3 section 6 steps 7-8).

    The agent's own suggestions (agent/music_agent.py) already carry both.
    The rule-engine's (raaga/selection.py) only carry a rationale string and a
    raw match ``score`` with no fixed ceiling, so when confidence is missing
    it is derived from the score: a soft, capped mapping rather than a
    hard-coded number, documented here because there is nowhere else a reader
    would know to look for it.
    """
    if not getattr(s, "reason", ""):
        s.reason = getattr(s, "rationale", "") or ""
    if getattr(s, "confidence", None) is None:
        score = float(getattr(s, "score", 0.0))
        s.confidence = round(min(0.95, 0.3 + 0.15 * score), 2)


@dataclass
class RenderedAudio:
    kind: str
    audio: np.ndarray
    sample_rate: int
    path: str = ""
    created_at: float = field(default_factory=time.time)
    duration: float = 0.0


class AppController:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.load()
        setup_logging(self.settings.log_level)
        self.store = ProjectStore(self.settings)
        self.jobs = JobManager(max_workers=3)
        self.undo = UndoManager()
        self.playback = PlaybackEngine(self.settings.sample_rate)
        self.voices = VoiceProfileManager()
        self.raagas = raaga_library()
        self.context = ConversationContext()
        self.voice_input = VoiceInputManager(self.settings)
        self.providers = provider_registry.build(
            self.settings, stt_name=self.voice_input.adapter.status())
        # The Knowledge Base: the permanent learned memory.  Opened, never
        # recreated - if this fails the application still runs, but it says so
        # rather than carrying on with an empty one that looks like loss.
        #
        # Opened *before* the agent, because the agent is given it: what the
        # agent hears has to reach the permanent memory, and it can only do
        # that if it was handed the Knowledge Base when it was built.
        self.kb: Optional[KnowledgeBaseService] = None
        self.knowledge_context: Optional[KnowledgeContextBuilder] = None
        try:
            path = getattr(self.settings, "knowledge_base_db", "") or None
            self.kb = KnowledgeBaseService.initialize_if_needed(
                Path(path) if path else None)
            self.knowledge_context = KnowledgeContextBuilder(self.kb)
        except Exception as exc:  # noqa: BLE001
            log.error("the Knowledge Base could not be opened: %s. Nothing "
                      "has been deleted; learned knowledge is untouched.", exc)
            self.kb = None

        # The musician behind the instrument: permanent memory, a curriculum
        # and everything it has learned so far.
        self.agent = MusicAgent(self.settings, self.raagas,
                                llm=self.providers.llm, kb=self.kb)

        # After the agent, because the migration reads the agent's own
        # repository to bring what it already knows into the Knowledge Base.
        if self.kb is not None:
            try:
                self._migrate_knowledge_base()
            except Exception as exc:  # noqa: BLE001
                log.error("the Knowledge Base migration did not run: %s. "
                          "Nothing has been deleted.", exc)

        # The Training tab: search for material, approve it, learn from it.
        # It shares the agent's memory so what it learns reaches the composer,
        # and the Knowledge Base so it accumulates across runs.
        try:
            self.training = TrainingController(
                self.settings, self.raagas, agent_repo=self.agent.repo,
                curriculum=self.agent.curriculum, kb=self.kb,
                # What a studied source taught becomes lessons the agent can
                # be examined on, rather than sitting unread in a report.
                on_report=self._file_stated_lessons)
        except Exception as exc:  # noqa: BLE001 - never block startup on it
            log.warning("the training system is unavailable: %s", exc)
            self.training = None

        self.project: Project = Project()
        self.project_dir: Optional[Path] = None
        self.dirty = False
        self._last_autosave = time.time()
        self._renders: Dict[str, RenderedAudio] = {}
        #: The exact render currently sitting in the playback engine, as an
        #: object rather than a name.  ``_cache_render`` makes a new
        #: ``RenderedAudio`` every time, so identity distinguishes "this is
        #: the audition you are already hearing" from "this is a different
        #: audition that happens to also be called 'audition'".
        self._loaded_render: Optional[RenderedAudio] = None
        self.status_text = "Ready"
        self.selection: Optional[Tuple[float, float]] = None
        self._playhead = 0.0
        #: The span the creator last actually listened to, and the name of
        #: it.  A comparison that jumps back to the top of the song is not
        #: a comparison of what they were hearing.
        self._audition: Optional[Tuple[Tuple[float, float], str]] = None
        self.last_evaluation = None

        # The action status contract (v0.3 section 6.1).  ``actions`` holds
        # the latest status per action name; ``_action_queue`` carries
        # statuses raised on a background job's worker thread across to the
        # UI thread, the same way JobManager carries job results (see
        # ``core/jobs.py`` and ``pump`` below).
        self.actions: Dict[str, ActionStatus] = {}
        self._action_queue: "queue.Queue[ActionStatus]" = queue.Queue()
        #: Recognised speech, waiting to be acted on by whoever calls
        #: ``pump``.  Speech capture runs on its own thread and used to run
        #: the whole command pipeline from there - interpreting the phrase,
        #: changing the project, and refreshing the window - which crashed
        #: the application the first time anyone spoke to it.  Jobs have
        #: always come back through ``pump``; speech now uses the same door.
        # Bounded, deliberately.  A live microphone is an unbounded source
        # of work: on 2026-09-06 a conversation held near the machine filled
        # this queue faster than it could be drained - 104 phrases captured
        # against 89 interpreted in two minutes - and the application died
        # with the backlog still growing.  A queue fed by the room has to be
        # allowed to drop, and to say that it dropped.
        self._utterance_queue: "queue.Queue[str]" = queue.Queue(maxsize=8)
        #: Which song is open.  Bumped whenever the project is replaced, so
        #: anything queued or in flight can tell that the song it was asked
        #: about is not the song that would receive it.
        self._project_generation = 0
        #: Typed instructions, which are deliberate and must not be dropped.
        #: Overheard speech may be discarded when the room outruns the
        #: interpreter; something the creator sat and typed may not, and it
        #: should not wait behind eight phrases nobody addressed to us.
        self._typed_queue: "queue.Queue[str]" = queue.Queue(maxsize=64)
        #: How many phrases were discarded because the queue was full.
        self._utterances_dropped = 0
        #: The interpretation currently in flight, if any, and which
        #: request it belongs to.  A cancelled older request must not clear
        #: a newer one's marker: doing so let a third request start beside
        #: the second and supersede it on the same target, losing it.
        self._interpreting: Optional[str] = None
        self._interpreting_id: int = 0
        self._request_seq: int = 0
        #: Which vocal render is the current one.  A ticket describes the
        #: song, and two renders of the same unchanged song are identical
        #: to it - so the older one's queued completion still wrote its
        #: take and played its follow-on after a newer render was asked
        #: for.  What distinguishes them is which was asked for last.
        self._vocal_request: int = 0
        self.last_suggestions: List = []
        #: The brief ``last_suggestions`` were made for, so selection
        #: feedback is attached to what was actually asked.
        self.suggested_for: Optional[CreativeBrief] = None
        #: Which ranking ``last_suggestions`` came from.  Bumped whenever a
        #: ranking is stored *and* whenever one is thrown away, so anything
        #: that cached a suggestion's reasoning can tell that the reasoning
        #: it holds belongs to a ranking that no longer exists.
        self.suggestion_epoch = 0

        # UI callbacks
        self.on_project_changed: Optional[Callable[[], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_conversation: Optional[Callable[[], None]] = None
        self.on_render: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_action: Optional[Callable[[ActionStatus], None]] = None

        self.voice_input.on_final = self._on_transcript_final
        self.voice_input.on_partial = self._on_transcript_partial
        self.voice_input.on_barge_in = self._on_barge_in
        self.voice_input.on_state = lambda st: self._notify_conversation()

        self.new_project("Untitled Song", write=False)

    # ==================================================================
    # plumbing
    # ==================================================================
    @property
    def sample_rate(self) -> int:
        return int(self.settings.sample_rate)

    def status(self, text: str) -> None:
        self.status_text = text
        log.info("status: %s", text)
        if self.on_status:
            self.on_status(text)

    def error(self, where: str, message: str, fallback: str = "") -> None:
        log.error("%s: %s", where, message)
        self.project.errors.append(ErrorRecord(where=where, message=message,
                                               fallback=fallback))
        self.status(message)
        if self.on_error:
            self.on_error(message)

    # -- the action status contract (v0.3 section 6.1) ----------------------
    def _action(self, action: str, state: ActionState, phase: str = "",
                message: str = "", code: str = "", detail: str = "",
                deliver_now: bool = True, target: str = "",
                epoch: int = 0) -> ActionStatus:
        """Record one step of an action's progress.

        With ``deliver_now`` (the default) the status is handed to
        ``self.status``/``self.error``/``on_action`` immediately, on whatever
        thread called this - safe only when the caller is already the UI
        thread.  A background job's worker function must pass
        ``deliver_now=False``; the status is then queued and delivered by
        ``pump`` on the UI thread, the same pattern ``JobManager`` uses for
        job results (``core/jobs.py``).
        """
        status = ActionStatus(action=action, state=state, phase=phase,
                              message=message, code=code, detail=detail,
                              target=target, epoch=epoch)
        if self._action_is_stale(status):
            log.info("action %s -> %s dropped as stale (target=%s epoch=%d)",
                     action, state.value, target, epoch)
            return status
        if deliver_now:
            # Otherwise ``_drain_actions`` records it on the UI thread at the
            # moment it is delivered, so ``self.actions`` never runs ahead
            # of what the creator has been shown.
            self.actions[action] = status
        if state == ActionState.FAILED:
            log.error("action %s failed [%s]: %s (%s)", action, code,
                      message, detail)
        else:
            log.info("action %s -> %s: %s", action, state.value,
                     phase or message)
        if deliver_now:
            self._deliver_action(status)
        else:
            self._action_queue.put(status)
        return status

    def _deliver_action(self, status: ActionStatus) -> None:
        """Apply a status's visible side effects. UI thread only."""
        # A status carrying a diagnostic code - failed or only a recovered
        # warning - belongs in the project's error log too (section 54):
        # "no silent failure" covers a recovered failure as much as a fatal
        # one.  ``error`` already updates the status bar, so a plain status
        # update only happens on the branch that is not already an error.
        if status.state == ActionState.FAILED or status.code:
            self.error(status.action, status.message or status.phase,
                      fallback=status.detail)
        elif status.phase or status.message:
            self.status(status.text)
        if self.on_action:
            try:
                self.on_action(status)
            except Exception:  # noqa: BLE001
                log.error("on_action callback failed", exc_info=True)

    def _action_is_stale(self, status: ActionStatus) -> bool:
        """A status from a superseded or cancelled job run is stale."""
        if not status.target or not status.epoch:
            return False
        return self.jobs.current_epoch(status.target) != status.epoch

    def _drain_actions(self) -> None:
        """Deliver any action statuses queued from a background job.

        Staleness is checked again here, not only when the status was
        queued: the job that queued it may have been superseded between
        then and this pump.
        """
        while True:
            try:
                status = self._action_queue.get_nowait()
            except queue.Empty:
                break
            if self._action_is_stale(status):
                log.info("queued action %s -> %s dropped as stale", status.action,
                         status.state.value)
                continue
            self.actions[status.action] = status
            self._deliver_action(status)

    def _changed(self, action: str = "", description: str = "",
                 undoable: bool = True) -> None:
        self.dirty = True
        self.project.touch()
        if action:
            self.project.log_history(action, description)
            if undoable:
                self.undo.commit(self.project, description or action)
        self._sync_context()
        if self.on_project_changed:
            self.on_project_changed()

    def _notify_conversation(self) -> None:
        if self.on_conversation:
            self.on_conversation()

    @property
    def playhead(self) -> float:
        """Where "here" is.

        The transport owns the position while audio is loaded; before anything
        has been rendered the creator can still move the playhead on the
        timeline, and that position has to survive.
        """
        if self.playback.duration > 0:
            return self.playback.position
        return self._playhead

    def _sync_context(self) -> None:
        self.context.duration = self.project.duration
        self.context.sections = list(self.project.sections)
        self.context.selection = self.selection
        self.context.playhead = self.playhead

    def pump(self) -> None:
        """Called by the UI timer: deliver job results on the UI thread."""
        events = self.jobs.drain()
        # Job results before action statuses: a queued "completed" status was
        # built on the worker thread alongside the result ``drain`` just
        # handed to its ``on_done`` callback, so the project state the status
        # describes (e.g. ``last_suggestions``) is already in place by the
        # time ``on_action`` sees it.
        self._drain_actions()
        self._drain_utterances()
        self._sync_context()
        if events and self.on_project_changed:
            self.on_project_changed()
        self.maybe_autosave()

    # ==================================================================
    # project lifecycle
    # ==================================================================
    def song_work_ticket(self, section_ids: Sequence[str] = ()) -> Dict:
        """What a background job about the song has to still be true for.

        Written down when the work is submitted and checked again when it
        comes back.  Everything between those two moments belongs to the
        creator: they can start another song, rewrite the tune, or lock the
        very section being worked on, and a result that ignores any of that
        overwrites a decision they made while waiting.

        Every job that changes the song takes one of these.  Guarding them
        one at a time is how the same fault reappeared three times in a day
        - the interpretation path was fixed and the lyric path, written
        afterwards, had the identical hole.
        """
        melody = self.project.melody()
        return {"generation": self._project_generation,
                "melody_version": melody.version if melody else None,
                "sections": tuple(section_ids)}

    def _merge_sections(self, draft: LyricsVersion,
                        chosen: Sequence[str]) -> LyricsVersion:
        """Take the chosen sections' lines from *draft*, the rest from now.

        A snapshot of the words taken when the job started is not enough,
        and no version number will catch it: edit_lyric_line changes a
        lyric version in place, so a creator who fixes a Pallavi line while
        the Charanam is being written leaves every number identical and the
        arriving draft still carries their old Pallavi text.  So the draft
        is not committed as a whole - only the lines it was actually asked
        to write are taken from it, and everything else is whatever is
        there when it lands.
        """
        current = self.project.lyrics_version()
        if current is None or not chosen:
            return draft
        wanted = set(chosen)
        lines = []
        for i, line in enumerate(draft.lines):
            existing = current.lines[i] if i < len(current.lines) else None
            if existing is None:
                lines.append(line)
                continue
            # A line the creator locked while waiting is their decision,
            # even inside the section they asked to have rewritten.
            if existing.section_id in wanted and not existing.locked:
                lines.append(line)
            else:
                # Copied, not shared.  Appending the object itself put the
                # same LyricLine in two versions, so editing a line in the
                # current version silently rewrote the older one that was
                # supposed to be a record of what it had said.
                lines.append(copy.deepcopy(existing))
        merged = replace(draft, lines=lines)
        return merged

    @staticmethod
    def lyric_fingerprint(lyrics) -> str:
        """What the words actually are, not which version they are.

        edit_lyric_line changes a lyric version in place, so its number is
        unchanged after an edit and a render made from the old words passes
        any check that compares numbers.  This compares the words.
        """
        if lyrics is None:
            return ""
        parts = [f"{line.id}:{line.text}:{int(bool(line.locked))}:"
                 f"{','.join(str(i) for i in line.note_indices)}"
                 for line in lyrics.lines]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def stale_reason(self, ticket: Dict) -> str:
        """Why a finished job must not be committed, or "" when it may be."""
        if ticket.get("generation") != self._project_generation:
            return "the song changed while I was working"
        melody = self.project.melody()
        expected = ticket.get("melody_version")
        if expected is not None:
            if melody is None:
                return "the tune it was written for is gone"
            if melody.version != expected:
                return "the tune changed while I was working"
        expected_words = ticket.get("lyric_fingerprint")
        if expected_words is not None:
            if self.lyric_fingerprint(self.project.lyrics_version()) != expected_words:
                return "the words changed while I was working"
        expected_voice = ticket.get("voice_profile_id")
        if expected_voice is not None and self.current_voice().id != expected_voice:
            return "you chose a different voice while I was working"
        known = {sec.id: sec for sec in (melody.sections if melody else ())}
        for sid in ticket.get("sections") or ():
            section = known.get(sid)
            if section is None:
                return "that section is no longer part of the tune"
            if section.locked:
                return f"{section.name} was locked while I was working"
        return ""

    def _forget_the_previous_song(self) -> None:
        """Drop what described the song we are leaving.

        Found by looking for more of the shape Arya kept finding rather
        than by another failure: state set while one song is open and read
        while another is.  last_evaluation is the one that mattered - it is
        the agent's judgement of a particular tune, and record_lessons
        attaches it to whatever project is open when the creator next says
        something.  Praise for one song became a lesson filed against
        another, in the agent's own learning.
        """
        self.last_evaluation = None
        self._loaded_render = None
        self._playhead = 0.0
        self._audition = None

    def _abandon_pending_requests(self, what: str) -> int:
        """Drop conversation still owed to a song we are leaving.

        Applying it to the new song would edit a song nobody was talking
        about.  This is the same fault as a ranking outliving its project,
        which is why the answer is the same one: the request carries the
        generation it was made in, and a generation that has moved on is a
        request that no longer has a subject.
        """
        self._project_generation += 1
        dropped = 0
        for q in (self._typed_queue, self._utterance_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                dropped += 1
        # Whatever is already interpreting belongs to the previous song too.
        self.jobs.cancel_target("voice")
        self.jobs.cancel_target("typed")
        if self._interpreting is not None:
            dropped += 1
        self._interpreting = None
        self._interpreting_id = 0
        if dropped:
            self.status(f"{what}: {dropped} unfinished request(s) were about "
                        f"the previous song, so I have let them go.")
        return dropped

    def _clear_ranking(self) -> None:
        """Forget the suggestions and the brief they were ranked for.

        A ranking belongs to one project's brief.  Carrying it across a
        project change left the panel able to say "the current brief
        suggested this" about a brief belonging to a song the creator had
        closed - a recommendation from a prior context presented as current
        evidence.  The controller owns the ranking, so the controller
        discards it; the panel reads the epoch and follows.
        """
        self.last_suggestions = []
        self.suggested_for = None
        self.suggestion_epoch += 1

    def new_project(self, title: str = "Untitled Song", write: bool = True) -> Project:
        self.playback.stop()
        self._renders.clear()
        if write:
            self.project, directory = self.store.create(title)
            self.project_dir = directory
        else:
            self.project = Project(title=title)
            self.project_dir = None
        self.project.voice_profile_id = self.voices.default().id
        self.selection = None
        self._clear_ranking()
        self._forget_the_previous_song()
        self._abandon_pending_requests("New song")
        self.dirty = not write
        self.undo.reset(self.project, "new project")
        self.context = ConversationContext()
        self._sync_context()
        self.status(f"New project: {self.project.title}")
        if self.on_project_changed:
            self.on_project_changed()
        return self.project

    def open_project(self, directory: Path) -> Project:
        self.playback.stop()
        self._renders.clear()
        self.project = self.store.open(Path(directory))
        self.project_dir = Path(directory)
        if not self.voices.get(self.project.voice_profile_id):
            self.project.voice_profile_id = self.voices.default().id
        self.selection = None
        self._clear_ranking()
        self._forget_the_previous_song()
        self._abandon_pending_requests("Opened another song")
        self.dirty = False
        self.undo.reset(self.project, "opened")
        self.context = ConversationContext()
        self._restore_renders()
        self._sync_context()
        self.status(f"Opened {self.project.title}")
        if self.on_project_changed:
            self.on_project_changed()
        return self.project

    def save(self) -> Optional[Path]:
        if self.project_dir is None:
            directory = self.store.projects_dir / \
                f"{self.project.title.replace(' ', '-').lower()}_{self.project.project_id[-6:]}"
            self.project_dir = self.store.ensure_dirs(directory)
        path = self.store.save(self.project, self.project_dir)
        self.store.register(self.project, self.project_dir)
        self.dirty = False
        self._last_autosave = time.time()
        self.status(f"Saved to {self.project_dir}")
        return path

    def save_as(self, directory: Path) -> Path:
        """Save under a new name - which is also how a song is renamed.

        The folder the creator picks *is* the name, the way Save As names a
        document anywhere else, so there is no separate rename and no name
        field on screen.  The song and the project share the one name.
        """
        target = self.store.save_as(self.project, self.project_dir, Path(directory))
        self.project_dir = target
        chosen = Path(directory).name.strip()
        if chosen and chosen != self.project.title:
            self.project.title = chosen
            self.project.brief.title = chosen
            self.store.save(self.project, target)
        self.dirty = False
        self.status(f"Saved as {self.project.title}")
        if self.on_project_changed:
            self.on_project_changed()
        return target

    def maybe_autosave(self) -> None:
        if not self.dirty or self.project_dir is None:
            return
        if time.time() - self._last_autosave < max(5, self.settings.autosave_seconds):
            return
        try:
            self.store.save(self.project, self.project_dir)
            self._last_autosave = time.time()
            self.dirty = False
            log.debug("autosaved")
        except Exception as exc:  # noqa: BLE001
            self.error("autosave", f"Autosave failed: {exc}")

    def recent_projects(self) -> List[Dict[str, str]]:
        return self.store.recent()

    def close(self) -> None:
        try:
            if self.dirty and self.project_dir:
                self.store.save(self.project, self.project_dir)
        except Exception as exc:  # noqa: BLE001
            log.error("final save failed: %s", exc)
        self.voice_input.close()
        self.playback.close()
        self.jobs.shutdown()
        try:
            if getattr(self, "kb", None) is not None:
                self.kb.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("closing the Knowledge Base failed: %s", exc)
        try:
            if getattr(self, "training", None) is not None:
                self.training.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("closing the training system failed: %s", exc)
        try:
            self.agent.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("closing the agent failed: %s", exc)

    # ==================================================================
    # the Knowledge Base
    # ==================================================================
    def _migrate_knowledge_base(self) -> None:
        """Bring existing knowledge in, once - knowledge spec section 47.

        Idempotent, and it never runs destructively: everything goes through
        the same duplicate control as any other write, so a second pass adds
        nothing rather than doubling anything.
        """
        if self.kb is None:
            return
        if self.kb.store.get_meta("migrated_existing_stores"):
            return
        from .kb.migrate import migrate_all

        training_db = getattr(self.settings, "training_db", "") or None
        report = migrate_all(
            self.kb, raagas=self.raagas,
            training_db=Path(training_db) if training_db else None,
            agent_repo=self.agent.repo)
        log.info("Knowledge Base migration: %s", report.summary())

    def knowledge_for(self, task: str, raaga: str = "", **kwargs):
        """What the Knowledge Base holds for a piece of work - section 19.

        This is how composition reads the Knowledge Base: through the service
        and the context builder, taking the smallest sufficient set rather than
        everything about the raga.
        """
        if self.knowledge_context is None:
            return None
        return self.knowledge_context.build(task, raga=raaga, **kwargs)

    def knowledge_health(self):
        """Section 40, for the UI and for a person asking."""
        if self.kb is None:
            return None
        from .kb.librarian import Librarian

        return Librarian(self.kb).health()

    def explain_knowledge(self, knowledge_id: str):
        """Section 41 - where a piece of knowledge came from."""
        return self.kb.provenance(knowledge_id) if self.kb else {}

    # ==================================================================
    # creative brief and raaga
    # ==================================================================
    def update_brief(self, **fields) -> None:
        brief = self.project.brief
        for key, value in fields.items():
            if hasattr(brief, key):
                setattr(brief, key, value)
        # The song title (v0.3 section 5) lives on the brief so it is part of
        # one creative statement, but the rest of the application - the
        # window title, the project panel, the saved project - reads
        # ``project.title``.  Only a non-empty title overwrites it, so an
        # untitled brief never blanks out a title set from the project panel.
        if brief.title.strip():
            self.project.title = brief.title.strip()
        self.project.current_stage = Stage.RAAGA if brief.summary() else Stage.BRIEF
        self._changed("brief.update", "Updated the creative brief")

    @staticmethod
    def _brief_is_describable(brief: CreativeBrief) -> bool:
        """Section 6 step 2: at least one of the three creative-intent
        fields must say something before anything can be suggested from it."""
        return bool(brief.situation.strip() or brief.mood.strip()
                   or brief.feel.strip())

    def _apply_kb_signal(self, brief: CreativeBrief, suggestions: List) -> None:
        """Query the learned Knowledge Base (section 6 step 4).

        ``agent.suggest_raagas`` (agent/music_agent.py) already answers from
        the agent's own learned-fact repository (``agent.repo``).  That is
        not the same store as ``self.kb`` - the newer, unified Knowledge Base
        that Training/Learn writes to (see ``raagacomposer/kb``) - and the
        agent does not consult it.  Rather than rewire the agent's ranking,
        this is the smallest hook that lets the KB speak too: for each
        shortlisted raaga it asks what the KB has learned about its mood
        (``rasa``), and nudges score/confidence up when a learned claim
        corroborates the brief.  A raaga the KB knows nothing about is left
        exactly as the agent or the rule engine scored it - this never
        invents a KB opinion that is not there.
        """
        if self.knowledge_context is None or not suggestions:
            return
        words = set(expand_feel_words(brief.mood, brief.feel, brief.situation,
                                      brief.notes))
        if not words:
            return
        touched = False
        for s in suggestions:
            try:
                context = self.knowledge_context.build(
                    "teach", raga=s.name, record_usage=False)
            except Exception as exc:  # noqa: BLE001 - the KB is optional here
                log.warning("Knowledge Base lookup failed for %s: %s",
                           s.name, exc)
                continue
            matched = [item for item in context.items
                      if item.predicate == "rasa"
                      and any(w in item.object_value.lower() for w in words)]
            if not matched:
                continue
            best = max(matched, key=lambda i: i.confidence)
            s.score = float(getattr(s, "score", 0.0)) + 0.4
            s.confidence = round(min(0.97, float(getattr(s, "confidence", 0.5))
                                     + 0.1), 2)
            note = (f"the Knowledge Base has learned {s.name} carries "
                   f"{best.object_value}")
            evidence = getattr(s, "evidence", None)
            if isinstance(evidence, list):
                evidence.append(note)
            s.reason = f"{getattr(s, 'reason', '')} ({note})".strip()
            touched = True
        if touched:
            suggestions.sort(key=lambda s: (-float(getattr(s, "score", 0.0)),
                                            s.name))

    def _run_apply_brief_pipeline(
            self, brief: CreativeBrief, limit: int, deliver_now: bool,
            ctx: Optional[JobContext] = None,
            epoch: int = 0) -> Tuple[ActionStatus, List]:
        """The whole of Apply Brief (section 6 steps 3-8), phases included.

        Runs identically whether called inline (``apply_brief_sync``, or a
        test) or from a background job's worker function (``apply_brief``);
        ``deliver_now`` controls whether each phase is delivered straight
        away or queued for ``pump`` to hand to the UI thread - see
        ``AppController._action``.  Never mutates ``self.project``: it
        returns the final status and the ranked suggestions, and it is the
        caller's job to write them into project state on the UI thread.
        """
        if ctx is not None:
            epoch = ctx.epoch
        # Words the engine cannot read are noted and then stepped over: the
        # ranking runs on what was understood, and the rest is worked out
        # afterwards (Arya's specification, 2026-09-06 13:16).  Anything
        # already worked out is substituted in first, so a resolution
        # reaches every reader of the brief at once.
        deferred = self.note_unreadable_words(brief)
        readings = self.readings_in_use(brief)
        brief = self.readable_brief(brief)
        try:
            self._action("apply_brief", ActionState.WORKING,
                        phase="Analyzing creative brief...",
                        deliver_now=deliver_now,
                        target="brief", epoch=epoch)
            if ctx:
                ctx.check()

            agent_failed = False
            try:
                suggestions = self.agent.suggest_raagas(brief, limit)
            except Exception as exc:  # noqa: BLE001 - fall back, never empty
                agent_failed = True
                log.error("the agent could not suggest raagas: %s", exc,
                         exc_info=True)
                self._action(
                    "apply_brief", ActionState.WORKING,
                    phase=f"The agent could not answer ({exc}); using the "
                          f"shipped raaga library instead.",
                    code="BRIEF-003", detail=repr(exc),
                    deliver_now=deliver_now,
                        target="brief", epoch=epoch)
                suggestions = suggest_raagas(brief, self.raagas, limit=limit)
            for s in suggestions:
                _normalize_suggestion(s)

            if ctx:
                ctx.check()
            self._action("apply_brief", ActionState.WORKING,
                        phase="Searching learned raga knowledge...",
                        deliver_now=deliver_now,
                        target="brief", epoch=epoch)
            self._apply_kb_signal(brief, suggestions)

            if ctx:
                ctx.check()
            self._action("apply_brief", ActionState.WORKING,
                        phase="Ranking suggestions...",
                        deliver_now=deliver_now,
                        target="brief", epoch=epoch)
            suggestions.sort(key=lambda s: (-float(getattr(s, "score", 0.0)),
                                            s.name))
            suggestions = suggestions[:limit]

            # An LLM re-rank is an enhancement, never a requirement: its
            # failure is logged and reported as a warning phase, not
            # swallowed (spec section 53's "never let stale output overwrite
            # newer intent" cousin here is "never let an optional step erase
            # a working result").
            llm = self.providers.llm
            if llm is not None and llm.available:
                try:
                    names = [s.name for s in suggestions] or \
                        self.raagas.names()[:12]
                    extra = llm.suggest_raagas(brief, names)
                    order = {str(e.get("raaga", "")).lower():
                             str(e.get("reason", "")) for e in extra}
                    for item in suggestions:
                        gloss = order.get(item.name.lower())
                        if gloss:
                            # Explanation integrity: the score and the reason
                            # are derived from the block map (raaga/emotion.py)
                            # and a model never replaces that derivation.  It
                            # may add a sentence beside it, attributed, so a
                            # creator can tell which half a claim came from.
                            item.reason = f"{item.reason} The adviser adds: {gloss}"
                    suggestions.sort(key=lambda s: (
                        s.name.lower() not in order,
                        -float(getattr(s, "score", 0.0))))
                except Exception as exc:  # noqa: BLE001
                    log.warning("LLM raaga advice failed: %s", exc,
                               exc_info=True)
                    self._action(
                        "apply_brief", ActionState.WORKING,
                        phase="AI re-ranking was unavailable; using the "
                              "ranked list from the raaga library and what "
                              "has been learned so far.",
                        code="BRIEF-004", detail=repr(exc),
                        deliver_now=deliver_now,
                        target="brief", epoch=epoch)

            if not suggestions:
                status = self._action(
                    "apply_brief", ActionState.FAILED,
                    message="No raaga could be suggested from this brief.",
                    code="BRIEF-002", deliver_now=deliver_now,
                        target="brief", epoch=epoch)
                return status, suggestions

            top = suggestions[0]
            plural = "s" if len(suggestions) != 1 else ""
            message = f"{len(suggestions)} raaga{plural} suggested; " \
                     f"{top.name} first."
            # Say what did not reach the ranking.  Claiming every word of
            # the brief was used when some of it was set aside is the thing
            # the specification names outright: "do not claim that all
            # requested moods influenced the present result if some were
            # omitted."
            if deferred:
                message += (f" Still working out {', '.join(deferred)} - "
                            f"not used yet.")
            # A guess that is shaping the result says so.  It used to
            # succeed silently: the deferral line simply disappeared and
            # nothing replaced it, so a creator could not tell that a
            # model's reading of their word had done the ranking.
            for term, words in readings:
                message += (f" Reading {term} as {' and '.join(words)} "
                            f"(unconfirmed).")
            if agent_failed:
                message += " (the agent was unavailable; used the shipped " \
                          "raaga library)"
            status = self._action("apply_brief", ActionState.COMPLETED,
                                  message=message, deliver_now=deliver_now,
                        target="brief", epoch=epoch)
            return status, suggestions
        except JobCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - section 6 step 11
            log.error("apply_brief failed: %s", exc, exc_info=True)
            status = self._action(
                "apply_brief", ActionState.FAILED,
                message=f"Applying the brief failed: {exc}",
                code="BRIEF-002", detail=repr(exc), deliver_now=deliver_now,
                        target="brief", epoch=epoch)
            return status, []

    def apply_brief(self, **fields) -> ActionStatus:
        """Apply Brief (section 6): validate, then rank in the background.

        Returns immediately with the STARTING status; the phases and the
        final COMPLETED/FAILED status arrive through ``on_action`` once
        ``pump`` drains them, so a slow LLM call never blocks the GUI
        (section 53).  A second call supersedes the first: both share the
        ``"brief"`` job target.
        """
        self.update_brief(**fields)
        brief = self.project.brief
        if not self._brief_is_describable(brief):
            return self._action(
                "apply_brief", ActionState.FAILED,
                message="Describe the situation, the mood or the feel "
                        "before applying the brief.",
                code="BRIEF-001")
        # A snapshot, not a reference: ``self.project.brief`` is mutable and
        # a second Apply Brief before this job runs would otherwise rewrite
        # this job's input out from under it (on top of the epoch-based
        # staleness check in JobManager, which only protects the *result*).
        brief_snapshot = copy.deepcopy(brief)

        def work(ctx: JobContext):
            # The worker labels everything it reports with the epoch on its
            # own context rather than reading a shared counter, so a run
            # superseded while it is still on the pool cannot borrow the
            # newer run's epoch.
            return self._run_apply_brief_pipeline(
                brief_snapshot, limit=4, deliver_now=False, ctx=ctx)

        brief_ticket = {"generation": self._project_generation, "sections": ()}
        job = self.jobs.submit(
            "brief.apply", "brief", work,
            on_done=lambda r: self._apply_brief_done(r, brief_ticket),
            on_error=self._apply_brief_job_error,
            on_cancelled=lambda: self._apply_brief_cancelled(job.epoch),
            description="Apply the creative brief")
        # Submitting bumped the epoch for "brief"; every status this run
        # raises - STARTING included - carries it, so a status from an
        # earlier run that is still in the queue is recognised as stale and
        # dropped instead of overwriting this run's words.
        return self._action("apply_brief", ActionState.STARTING,
                            phase="Applying the brief...",
                            target="brief", epoch=job.epoch)

    def _apply_brief_cancelled(self, epoch: int) -> None:
        """The creator cancelled the work (section 6.1's Cancelled state).

        A superseded run also arrives here, because JobManager reports
        "stale" through the same callback.  The two are told apart by what
        the creator has already been shown: a newer run has recorded its
        STARTING with a higher epoch, a plain cancel has not.  Cancelling
        bumps the epoch as well, so the CANCELLED status is deliberately
        not tagged with one - it is the last word on this run, not a report
        from inside it.
        """
        current = self.actions.get("apply_brief")
        if current is not None and current.epoch > epoch:
            return
        self._action("apply_brief", ActionState.CANCELLED,
                     message="Applying the brief was cancelled.")

    def apply_brief_sync(self, **fields) -> ActionStatus:
        """Apply Brief, run inline and returned - for tests and the
        conversational path (spec sections 6, 20), where a synchronous
        answer is simpler than waiting on a callback."""
        self.update_brief(**fields)
        brief = self.project.brief
        if not self._brief_is_describable(brief):
            return self._action(
                "apply_brief", ActionState.FAILED,
                message="Describe the situation, the mood or the feel "
                        "before applying the brief.",
                code="BRIEF-001")
        self._action("apply_brief", ActionState.STARTING,
                    phase="Applying the brief...")
        status, suggestions = self._run_apply_brief_pipeline(
            brief, limit=4, deliver_now=True)
        if status.state == ActionState.COMPLETED:
            self._store_brief_suggestions(suggestions)
        return status

    def _store_brief_suggestions(self, suggestions: List) -> None:
        """Write a completed Apply Brief's results into project state.  UI
        thread only - see the threading rule in this module's docstring."""
        self.last_suggestions = suggestions
        self.suggestion_epoch += 1
        self.project.raaga.alternatives = [s.name for s in suggestions]
        # The brief these answer, kept as it was at the time.  Selection
        # feedback has to be attached to the brief the suggestions were made
        # for, not to whatever is in the panel when the creator gets round to
        # choosing: apply a brief, apply a second one, then pick from the
        # first list, and the agent would otherwise learn that the raaga
        # suits a feeling nobody was asking about.
        self.suggested_for = replace(self.project.brief)

    def _apply_brief_done(self, result: Tuple[ActionStatus, List],
                          ticket: Optional[Dict] = None) -> None:
        status, suggestions = result
        stale = self.stale_reason(ticket) if ticket else ""
        if stale:
            # A ranking written for the song we left would become the new
            # song's alternatives and the brief its feedback is attached
            # to, which is the same fault in a tenth place.
            self.status(f"I did not keep those suggestions: {stale}.")
            return
        if status.state == ActionState.COMPLETED:
            self._store_brief_suggestions(suggestions)
            self.dirty = True
            self.project.touch()
            if self.on_project_changed:
                self.on_project_changed()

    def _apply_brief_job_error(self, exc: BaseException) -> None:
        # The pipeline catches its own exceptions and always returns a
        # status, so this is a safety net for a truly unexpected crash
        # inside the job machinery rather than the normal failure path.
        log.error("apply_brief job crashed unexpectedly: %s", exc,
                  exc_info=True)
        self._action(
            "apply_brief", ActionState.FAILED,
            message=f"Applying the brief failed unexpectedly: {exc}",
            code="BRIEF-002", detail=repr(exc))

    def note_unreadable_words(self, brief=None) -> List[str]:
        """Collect the brief's words the engine cannot read, and keep them.

        Deliberately not a gate.  The specification is explicit that an
        unfamiliar word must neither be discarded nor allowed to hold up
        the request, so this records and returns; the ranking proceeds on
        whatever was understood.
        """
        brief = brief or self.project.brief
        if self.agent is None:
            return []
        # Mood and feel only.  Situation and notes are narrative - "a man on
        # a terrace late at night" - and scanning them reported *terrace* and
        # *man* as unreadable feelings, which is both noise and a promise to
        # investigate words that were never moods.  These two fields are
        # where the creator states a feeling, so they are where an unread
        # word is worth chasing.
        terms = vocabulary.unknown_terms(brief.mood, brief.feel,
                                         raagas=self.raagas)
        if not terms:
            return []
        try:
            return self.agent.repo.note_unknown_terms(terms)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record unreadable words: %s", exc)
            return []

    def readable_brief(self, brief=None):
        """The brief with settled meanings substituted in.

        A resolution earns its keep here: once "nervy" is known to mean
        "nervous", the words go into the text before anything scores it, so
        the emotion vector, the raaga tags and the tempo hint all improve
        at once without any of them knowing this feature exists.
        """
        brief = brief or self.project.brief
        if self.agent is None:
            return brief
        try:
            table = self.agent.repo.resolutions()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read resolved terms: %s", exc)
            return brief
        if not table:
            return brief
        clone = replace(
            brief,
            mood=vocabulary.resolve_text(brief.mood, table),
            feel=vocabulary.resolve_text(brief.feel, table),
            situation=vocabulary.resolve_text(brief.situation, table),
            notes=vocabulary.resolve_text(brief.notes, table))
        return clone

    def readings_in_use(self, brief=None) -> List[Tuple[str, List[str]]]:
        """Unconfirmed meanings this brief is actually relying on.

        Only the ones whose word appears in the brief: telling a creator
        about a guess that is not touching their song would be noise, and
        the point of saying it at all is that a guess is shaping *this*
        result.
        """
        brief = brief or self.project.brief
        if self.agent is None:
            return []
        try:
            readings = self.agent.repo.unconfirmed_readings()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read unconfirmed meanings: %s", exc)
            return []
        blob = f"{brief.mood} {brief.feel}".lower()
        return [(term, words) for term, words in readings
                if re.search(rf"\b{re.escape(term)}\b", blob)]

    def vocabulary_report(self) -> str:
        """What became of the words the engine could not read."""
        if self.agent is None:
            return "No agent, so nothing is being learned."
        terms = self.agent.repo.unresolved_terms(limit=100)
        if not terms:
            return "Every word of every brief so far was understood."
        rows = []
        for status in (vocabulary.NEEDS_USER_INPUT, vocabulary.PENDING,
                       vocabulary.INVESTIGATING, vocabulary.RESOLVED):
            group = [t for t in terms if t.status == status]
            if group:
                rows.append(f"{status}:")
                rows.extend(f"  {t.describe()}" for t in group)
        return "\n".join(rows)

    def investigate_unknown_words(self, limit: int = 3) -> List[str]:
        """Work out what the outstanding words mean, in the background.

        Runs after the brief has already been answered, which is the whole
        point: the creator waited for nothing.  Only terms that are still
        pending are picked up, and a term that has failed too often stops
        being retried and starts waiting for a person.
        """
        if self.agent is None:
            return []
        pending = self.agent.repo.unresolved_terms(
            status=vocabulary.PENDING, limit=limit)
        if not pending:
            return []
        llm = self.providers.llm
        done: List[str] = []

        def work(ctx: JobContext) -> List[Tuple[str, str]]:
            results = []
            for i, term in enumerate(pending):
                ctx.progress((i + 1) / len(pending),
                             f"Working out what {term.term!r} means")
                self.agent.repo.set_term_status(term.term,
                                                vocabulary.INVESTIGATING)
                mapped, confidence, evidence = vocabulary.investigate(
                    term.term, llm)
                status = self.agent.repo.record_investigation(
                    term.term, mapped, confidence, evidence)
                results.append((term.term, status))
            return results

        def finished(results: List[Tuple[str, str]]) -> None:
            settled = [t for t, s in results if s == vocabulary.RESOLVED]
            asking = [t for t, s in results if s == vocabulary.NEEDS_USER_INPUT]
            parts = []
            if settled:
                parts.append(f"worked out {', '.join(settled)}")
            if asking:
                parts.append(f"could not work out {', '.join(asking)} - "
                             f"I will need you for those")
            self.status("; ".join(parts) or "no new words were settled")

        self.jobs.submit("vocabulary.investigate", "vocabulary",
                         work, on_done=finished,
                         on_error=lambda e: log.warning(
                             "investigating words failed: %s", e),
                         description="Work out unfamiliar mood words")
        return [t.term for t in pending]

    def raaga_suggestions(self, limit: int = 4) -> List:
        """"Suggest from the brief" (section 7): reruns the Apply Brief
        ranking inline against the current brief, without touching it or
        requiring the situation/mood/feel validation - the brief is already
        applied by the time this button is reachable."""
        brief = self.project.brief
        status, suggestions = self._run_apply_brief_pipeline(
            brief, limit=limit, deliver_now=True)
        if status.state == ActionState.COMPLETED:
            self._store_brief_suggestions(suggestions)
        return suggestions

    def _file_stated_lessons(self, report) -> None:
        """A completed source becomes lessons; never a reason to fail a run."""
        if self.agent is None:
            return
        try:
            made = self.agent.file_stated_lessons(report)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not file lessons from the report: %s", exc)
            return
        if made:
            self.status(f"{len(made)} lesson(s) to be examined on from "
                        f"{getattr(report.source, 'title', 'the source')}")

    def _feedback_brief(self, name: str) -> Optional[CreativeBrief]:
        """The brief this raaga was suggested for, if we actually know.

        Only a choice made *among the suggestions we offered* is feedback
        about a feeling: that is the one case where the raaga and the brief
        are known to belong together.  Naming a raaga that is not in the
        current list is an override, not a preference - we do not know which
        brief the creator had in mind, and guessing taught the agent that a
        raaga picked for a grieving brief suits a wedding.
        """
        if self.suggested_for is None:
            return None
        if name not in (self.project.raaga.alternatives or []):
            return None
        return self.suggested_for

    def select_raaga(self, name: str, rationale: str = "",
                     by_creator: bool = True) -> Raaga:
        """Choose the raaga to compose in.

        ``by_creator`` is what separates a choice from a default.  A creator
        picking one of the suggestions is a training signal the pack asks us
        to learn from (document 05 section 6); the application picking one
        for itself because nobody had, in ``require_raaga``, is not - and
        counting it as one would have the agent learning its own habits back
        from itself.
        """
        raaga = self.raagas.get(name)
        if raaga is None:
            raise KeyError(f"Unknown raaga: {name}")
        if self.project.raaga.locked:
            raise LockedContentError(
                f"The raaga is locked to {self.project.raaga.selected}. Unlock to change it.")
        offered = list(self.project.raaga.alternatives)
        self.project.raaga.selected = raaga.name
        self.project.raaga.rationale = rationale
        self.project.raaga.state = ApprovalState.APPROVED
        self.project.raaga.version += 1
        self.project.current_stage = Stage.TUNE
        self._changed("raaga.select", f"Selected raaga {raaga.name}")
        if by_creator and self.agent is not None:
            # Anything ranked above what they actually took was offered and
            # not taken; that is weaker evidence than a rejection and is
            # weighted as such.
            try:
                answered = self._feedback_brief(raaga.name)
                passed = offered[:offered.index(raaga.name)] \
                    if raaga.name in offered else []
                if answered is not None:
                    self.agent.record_raaga_choice(answered, raaga.name, passed)
            except Exception as exc:  # noqa: BLE001 - never block a selection
                log.warning("could not record the raaga choice: %s", exc)
        return raaga

    def reject_raaga(self, name: str, comment: str = "") -> Dict[str, float]:
        """The creator turned a suggestion down, and possibly said why.

        Returns the dimensions the comment moved, so a caller can show what
        was understood rather than silently absorbing it.
        """
        if self.agent is None:
            return {}
        answered = self._feedback_brief(name)
        if answered is None:
            # Turning down a raaga we did not suggest for a brief we cannot
            # identify is not something to learn from.
            log.info("no suggestion context for %s; nothing learned", name)
            return {}
        correction = self.agent.reject_raaga(answered, name, comment)
        detail = f"Noted: {name} is not right for this"
        if correction:
            detail += " (" + ", ".join(
                f"{'less' if v < 0 else 'more'} {d}"
                for d, v in sorted(correction.items())) + ")"
        self._changed("raaga.reject", detail)
        return correction

    def cast_lead(self):
        """Who plays the lead, and why - through the one shared policy.

        The tune itself is hummed now (specification 10.1), so this casts
        the audition and the arrangement, which used to decide separately
        and could disagree from the same brief.  ``music/casting.py`` holds
        the order; this supplies the controller's richer ranking, which
        asks a language model before the lexicon.
        """
        from .music import casting

        return casting.cast(
            "lead",
            preferred=self.project.brief.instruments_preferred,
            avoided=self.project.brief.instruments_avoided,
            feel_words=sorted(expand_feel_words(
                self.project.brief.mood, self.project.brief.feel,
                self.project.brief.situation, self.project.brief.notes)),
            configured=getattr(self.settings, "tune_instrument", ""),
            # ``suggest_instruments`` reads the brief's avoided list itself,
            # so the one the policy passes is dropped rather than passed
            # twice - and the policy checks it again on the result anyway.
            rank=lambda words, _avoid, role="lead", limit=3:
                self.suggest_instruments(words, role=role, limit=limit))

    def tune_instrument(self):
        """The instrument alone, for callers that do not need the reason."""
        return self.cast_lead().instrument

    def audition_raaga(self, name: str = "", play: bool = True):
        """Play a raaga's exact arohanam and avarohanam (pack section 7).

        The audition step between suggesting a raaga and composing in one:
        the creator hears the scale and confirms it is the raaga they meant.
        What is played is exactly what the library stores - no phrase, no
        ornament, no chosen register - so that a disagreement is about the
        raaga rather than about the performance.

        Returns the plan whether or not there is an audio device, because
        what was *meant* to be played is the checkable part and a machine
        with no sound card should still be able to say what it would sound.
        """
        raaga = self.raagas.get(name) if name else self.current_raaga()
        if raaga is None:
            self.status("Choose a raaga to hear first.")
            return None

        plan = audition.plan(raaga, tonic=self.project.raaga.tonic_midi
                             or audition.TONIC)
        if not audition.is_playable(plan):
            # Pack document 01 section H rule 7: an audition that collapses
            # into one repeated note has not demonstrated anything.
            self.error("audition",
                       f"{raaga.name} has no scale to play - its arohanam or "
                       f"avarohanam is missing from the library.")
            return plan

        instrument = self.tune_instrument()
        audio = render_notes(plan.notes, instrument, self.sample_rate,
                             total_seconds=plan.seconds)
        self._cache_render("audition", audio)
        self.status(f"{raaga.name}: {len(plan.ascending)} swaras up, "
                    f"{len(plan.descending)} down")
        if play:
            self.play_render("audition")

        # The pack counts hearing a raaga as a weak signal in its favour
        # (document 05 section 6, auditioned +0.2) - weaker than choosing it,
        # because listening is not yet agreeing.  Only when we know which
        # brief it answers, on the same rule selection feedback follows.
        answered = self._feedback_brief(raaga.name)
        if answered is not None and self.agent is not None:
            try:
                self.agent.audition_raaga(answered, raaga.name)
            except Exception as exc:                             # noqa: BLE001
                log.warning("could not record the audition: %s", exc)
        return plan

    def set_raaga_lock(self, locked: bool) -> None:
        self.project.raaga.state = ApprovalState.LOCKED if locked else ApprovalState.APPROVED
        self._changed("raaga.lock", f"{'Locked' if locked else 'Unlocked'} the raaga")

    def current_raaga(self) -> Optional[Raaga]:
        return self.raagas.get(self.project.raaga.selected)

    def require_raaga(self) -> Raaga:
        """The raaga to compose in, choosing one if the creator has not.

        Apply Brief ranks by emotional fit and says so - that is the Stage 1
        pack's engine and it does not care what the agent can play.  This is
        a different question.  Nobody has chosen anything and a tune is about
        to be written, so among the raagas that fit, prefer one there is
        something to compose *with*: a raaga somebody curated prayogas,
        resting notes and gamaka for, or better, one the agent has studied.
        A bare parent scale is a last resort here, not a first answer.
        """
        raaga = self.current_raaga()
        if raaga is not None:
            return raaga
        suggestions = self.raaga_suggestions(5)
        studied = set(self.agent.repo.known_raagas()) if self.agent else set()

        def playable(suggestion) -> tuple:
            entry = self.raagas.get(suggestion.name)
            if entry is None:
                return (2, 0)
            return (0 if entry.name in studied else 1 if not entry.scale_only
                    else 2, 0)

        # by_creator=False throughout: nobody chose this, the application did
        # because a tune was about to be written, and learning a preference
        # from it would be the agent learning its own habits back from itself.
        best = min(suggestions, key=playable) if suggestions else None
        if best is None:
            fallback = self.raagas.get("Mohanam") or self.raagas.all()[0]
            return self.select_raaga(fallback.name, "a safe default",
                                     by_creator=False)
        return self.select_raaga(best.name, best.rationale, by_creator=False)

    def composing_raaga(self) -> Raaga:
        """The raaga as the agent knows it: learned phrases included."""
        reference = self.require_raaga()
        try:
            learned, completeness = self.agent.raaga_for_composition(reference.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("falling back to the reference raaga: %s", exc)
            return reference
        if learned is None:
            return reference
        if completeness:
            log.info("composing in %s from memory (%.0f%% learned, %d phrases)",
                     learned.name, completeness * 100, len(learned.prayogas))
        return learned

    # ==================================================================
    # tune
    # ==================================================================
    def melody_options(self, seed: Optional[int] = None) -> MelodyOptions:
        raaga = self.require_raaga()
        profile = self.current_voice()
        brief = self.project.brief
        melody = self.project.melody()
        # One place decides the tempo, in this order: what the creator asked
        # for, then what the tune is already at, then what the brief and the
        # raaga suggest.  generate_tune used to overwrite this a line after
        # asking for it, which is how an explicit 200 bpm became 80 again on
        # the next Generate Tune.
        if brief.tempo_preference:
            tempo = int(brief.tempo_preference)
        elif melody is not None:
            tempo = melody.tempo_bpm
        else:
            tempo = infer_tempo(brief, raaga)
        return MelodyOptions(
            tempo_bpm=tempo,
            # The tune and the beat take their cycle from the same place,
            # which is what keeps them synchronised (specification 11.3).
            # A brief that names a tala wins over the tune's current cycle,
            # because choosing one is how a creator changes it.
            beats_per_cycle=self.current_tala().aksharas,
            tonic_midi=self.project.raaga.tonic_midi,
            voice_low=profile.range_low, voice_high=profile.range_high,
            intensity=0.6, seed=seed if seed is not None else int(time.time()) % 9999,
            song_type=brief.song_type, duration_target=brief.duration_target)

    def generate_tune(self, seed: Optional[int] = None) -> None:
        self.take_the_floor("generate tune")
        raaga = self.composing_raaga()
        opts = self.melody_options(seed)
        version = (max((m.version for m in self.project.melodies), default=0)) + 1
        self.status(f"Composing a tune in {raaga.name}...")

        brief = self.project.brief
        project_id = self.project.project_id
        max_rewrites = max(0, int(getattr(self.settings, "compose_rewrites", 3)))
        threshold = float(getattr(self.settings, "compose_threshold", 0.7))
        bank = self.agent.phrase_bank(raaga.name)

        def work(ctx: JobContext) -> MelodyVersion:
            ctx.progress(0.15, "Planning sections")
            # A section the creator named is not an optional repeat.  The
            # planner used to see only a template and a duration, so a
            # 60-second brief asking for an Anupallavi got a tune with no
            # Anupallavi and no explanation.
            wants = read_section_requests(brief.notes, brief.feel,
                                          brief.situation)
            plan_notes: List[str] = []
            sections = plan_sections(opts.duration_target, opts.tempo_bpm,
                                     opts.beats_per_cycle, opts.song_type,
                                     requested=wants.wanted,
                                     refused=wants.refused,
                                     notes=plan_notes)

            # What the raaga's lessons already say - critiques, failed
            # rewrites of an earlier tune, creator feedback - applies from
            # the first draft, the same way agent/practice.py always builds
            # guidance before an attempt runs rather than only after one
            # fails.
            initial_guidance = build_guidance(self.agent.repo, raaga.name)
            opts.guidance = initial_guidance
            guidance_note = initial_guidance.describe()

            ctx.progress(0.4, "Writing phrases")
            melody = melody_engine.generate(raaga, opts, sections, version=version)

            # A tune must earn its keep: the agent listens back with its own
            # evaluator (originality included) and, when it falls short,
            # rewrites with guidance built from what went wrong - three tries
            # by default (settings.compose_rewrites), best kept (spec section
            # 6.1's phase contract makes each rewrite visible rather than a
            # silent retry loop).
            best, best_score = melody, -1.0
            rewrite_lines: List[str] = []
            for attempt in range(max_rewrites + 1):
                ctx.check()
                ctx.progress(min(0.8, 0.45 + 0.08 * attempt), "Listening back")
                evaluation = self.agent.evaluator(raaga.name).evaluate(
                    melody.notes, raaga, brief=brief, tempo_bpm=opts.tempo_bpm,
                    expected_seconds=opts.duration_target, learned_phrases=bank)
                overall = evaluation.overall()
                original = (evaluation.originality is None
                           or evaluation.originality.is_original)
                if overall > best_score:
                    best, best_score = melody, overall

                if overall >= threshold and original:
                    break
                if attempt >= max_rewrites:
                    break

                try:
                    self.agent.record_lessons(
                        evaluation, raaga=raaga.name, task="composition",
                        method="generate", result=overall,
                        source_run=f"{project_id}:v{version}:try{attempt + 1}")
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not record lessons from this rewrite: %s", exc)

                kinds = sorted({f.kind for f in evaluation.findings})
                rewrite_lines.append(f"rewrite {attempt + 1}: {', '.join(kinds)}")

                guidance = build_guidance(self.agent.repo, raaga.name)
                opts.guidance = guidance
                opts.seed += 7919
                guidance_note = guidance.describe()
                ctx.progress(min(0.82, 0.5 + 0.08 * attempt),
                            f"Rewriting: {', '.join(guidance.kinds[:3])}"
                            if guidance.kinds else "Rewriting")
                melody = melody_engine.generate(raaga, opts, sections,
                                                version=version)

            ctx.progress(0.9, "Checking raaga fidelity")
            check = validate(best, raaga, opts.voice_low, opts.voice_high)
            best.validation = rewrite_lines + check.issues
            best.plan_notes = list(plan_notes)
            best.guidance_note = guidance_note
            return best

        # Composing replaces the tune, so the tune's own version is not a
        # precondition; which song is.
        ticket = {"generation": self._project_generation, "sections": ()}
        self.jobs.submit("tune.generate", "melody:all", work,
                         on_done=lambda m: self._tune_ready(m, "Generated",
                                                            ticket),
                         on_error=lambda e: self.error("tune", f"Tune generation failed: {e}"),
                         description=f"Compose a tune in {raaga.name}")

    # ==================================================================
    # beat (specification 11): percussion as its own layer
    # ==================================================================
    def current_tala(self):
        """The cycle this song is in.

        A melody records only ``beats_per_cycle``, so an existing tune says
        which tala it was written in by its beat count.  A brief may name
        one directly once there is a way to ask.
        """
        from .music import tala as tala_module

        return self.tala_choice().tala

    def tala_choice(self):
        """The cycle, and why - so the window can show it and be argued with.

        Order matters and is the same as everywhere else: what the creator
        asked for, then what the song is already in, then what the brief
        suggests.  Adding a tala to an existing tune keeps that tune's
        timing: the beat is written against this cycle, and the notes are
        not touched.
        """
        from .music import tala as tala_module

        named = tala_module.find(getattr(self.project.brief, "tala", ""))
        if named is not None:
            return tala_module.TalaChoice(named, "you asked for this cycle",
                                          True)
        melody = self.project.melody()
        if melody is not None:
            existing = tala_module.for_beats(melody.beats_per_cycle)
            return tala_module.TalaChoice(
                existing, "the tune is already in this cycle, and adding a "
                          "beat does not rewrite it")
        raaga = self.raagas.get(self.project.raaga.selected or "")
        return tala_module.suggest(self.project.brief, raaga)

    def generate_beat(self, density: str = "", seed: Optional[int] = None,
                      autoplay: bool = False) -> None:
        """Make a beat, without touching the tune (specification 11.9).

        The beat is written against the tala rather than the melody's
        notes, so the two line up by construction - same tempo, same
        cycle - and either can be replaced without disturbing the other.
        """
        from .music import beat as beat_engine
        from .music import tala as tala_module

        self.take_the_floor("generate beat")
        melody = self.project.melody()
        tala = self.current_tala()
        tempo = melody.tempo_bpm if melody else infer_tempo(
            self.project.brief, self.composing_raaga())
        duration = (melody.duration if melody
                    else float(self.project.brief.duration_target))
        if duration <= 0:
            self.status("Set a length or write a tune before making a beat.")
            return

        version = BeatVersion(
            version=len(self.project.beats) + 1,
            label="Beat", tala=tala.name, tempo_bpm=int(tempo),
            density=density or tala_module.DEFAULT_DENSITY,
            duration=float(duration),
            seed=seed if seed is not None else int(time.time()) % 9999)
        self.status(f"Laying down a beat in {tala.describe()}...")

        beat_ticket = self.song_work_ticket()

        def work(ctx: JobContext) -> BeatVersion:
            ctx.progress(0.4, f"{tala.name} at {version.tempo_bpm} bpm")
            return beat_engine.realise(version)

        self.jobs.submit(
            "beat.generate", "beat:all", work,
            on_done=lambda b: self._beat_ready(b, "Beat", autoplay,
                                               beat_ticket),
            on_error=lambda e: self.error("beat", f"Beat generation failed: {e}"),
            description=f"Lay down a {tala.name} beat")

    def beat_variation(self, strength: str = "moderate",
                       autoplay: bool = False) -> None:
        """A different take on the same beat - never a different tala."""
        from .music import beat as beat_engine

        self.take_the_floor("beat variation")
        previous = self.project.beat()
        if previous is None:
            return self.generate_beat(autoplay=autoplay)
        self.status(f"Varying the beat ({strength})...")

        variation_ticket = self.song_work_ticket()

        def work(ctx: JobContext) -> BeatVersion:
            ctx.progress(0.4, "Reworking the strokes")
            return beat_engine.realise(
                beat_engine.vary(previous, strength=strength))

        self.jobs.submit(
            "beat.variation", "beat:all", work,
            on_done=lambda b: self._beat_ready(b, "Beat variation", autoplay,
                                               variation_ticket),
            on_error=lambda e: self.error("beat", f"Beat variation failed: {e}"),
            description="Vary the beat")

    def _beat_ready(self, version: BeatVersion, what: str,
                    autoplay: bool = False,
                    ticket: Optional[Dict] = None) -> None:
        stale = self.stale_reason(ticket) if ticket else ""
        if stale:
            self.status(f"I did not keep that beat: {stale}.")
            return
        self.project.beats.append(version)
        self.project.approved_beat = version.version
        self.status(f"{what} v{version.version}: {version.summary()}")
        # The arrangement is updated before the undo point is taken, so one
        # Add Beat is one step.  It used to commit after appending the beat
        # and again after arranging it, and a single Undo then left the new
        # beat selected while the song still played the old one.
        self._beat_into_the_song(version, record=False)
        self._changed("beat.generate", f"{what} v{version.version}",
                      undoable=True)
        self.render_beat(autoplay=autoplay)

    def _beat_into_the_song(self, version, record: bool = True) -> None:
        """Make the beat part of what the song plays.

        Krish asked that adding a tala put percussion on the tune.  It laid
        the beat down and stopped: the full mix takes its parts from the
        arrangement, so a song with no arrangement mixed without the beat,
        and a song with one kept playing whichever beat happened to be in
        it when Auto Arrange last ran - v1, while v2 was selected.  Neither
        is discoverable; both look like the beat simply not working.

        Only the rhythm layer is touched.  Changing the beat is not a
        reason to rebuild the accompaniment or to overwrite anything the
        creator has locked.
        """
        melody = self.project.melody()
        if melody is None or not version.notes:
            return
        arrangement = self.project.arrangement()
        if arrangement is None:
            # Nothing to add to yet.  A rhythm layer on its own is what was
            # asked for; inventing the rest of an arrangement is not.
            arrangement = ArrangementVersion(
                version=max((a.version for a in self.project.arrangements),
                            default=0) + 1,
                label="Percussion")
            self.project.arrangements.append(arrangement)
        first_sung = next((s.start for s in melody.sections
                           if not s.kind.instrumental), 0.0)
        total = max(melody.duration, self.project.duration)
        said = arranger.apply_beat(arrangement, version,
                                   self.beat_instrument(), first_sung, total)
        self.status(said)
        if record:
            self._changed("arrange.beat", said)

    def render_beat(self, autoplay: bool = False) -> None:
        """Sound the beat on its own, so it can be judged on its own."""
        version = self.project.beat()
        if version is None or not version.notes:
            self.status("There is no beat to play yet.")
            return
        instrument = self.beat_instrument()
        sr = self.sample_rate
        provider = self.providers.music

        def work(ctx: JobContext) -> np.ndarray:
            ctx.progress(0.3, f"Sounding the beat on {instrument.name}")
            audio = provider.render_part(version.notes, instrument.key, sr,
                                         total_seconds=version.duration + 0.5,
                                         seed=version.seed)
            from .audio import dsp
            stereo = dsp.normalize_loudness(dsp.pan_mono(audio, 0.0), sr, -18.0)
            return dsp.limiter(stereo, -1.0, sr)

        beat_render_ticket = self.song_work_ticket()

        def done(audio: np.ndarray) -> None:
            stale = self.stale_reason(beat_render_ticket)
            if stale:
                self.status(f"I did not keep that beat audio: {stale}.")
                return
            path = self._write_artifact("audio", f"beat_v{version.version}.wav",
                                        audio)
            version.audio_path = str(path)
            self._cache_render("beat", audio, str(path))
            self.status(f"Beat ready - {version.summary()}")
            if autoplay:
                self.play_render("beat")

        self.jobs.submit("render.beat", "render:beat", work, on_done=done,
                         on_error=lambda e: self.error("beat",
                                                       f"Beat render failed: {e}"),
                         description="Render the beat")

    def beat_instrument(self):
        """Who plays the beat, through the one casting policy.

        A South Indian song is kept on a mridangam unless the creator says
        otherwise: ranking on feel alone put a tambourine under a Carnatic
        tune because "celebration" scores well on one, which is true of the
        word and wrong about the music.  The arrangement has always applied
        this rule; stating it here keeps the two agreeing.
        """
        from .music import casting

        brief = self.project.brief
        words = sorted(expand_feel_words(brief.mood, brief.feel,
                                         brief.situation, brief.notes))
        carnatic = ("carnatic" in " ".join(words).lower()
                    or brief.language.lower() in ("tamil", "telugu", "kannada",
                                                  "malayalam", "sanskrit"))
        preferred = list(brief.instruments_preferred)
        if carnatic:
            preferred.append("mridangam")
        return casting.cast(
            "rhythm", preferred=preferred,
            avoided=brief.instruments_avoided, feel_words=words,
            default="mridangam").instrument

    def _compose_in_named_raaga(self, cmd: Command) -> bool:
        """Honour a raaga named in a compose request.  False means do not.

        Selecting it rather than passing it through means the choice is
        recorded, undoable and visible in the panel - the same path the
        "set the raaga" command already takes - so the tune and what the
        window says about it cannot disagree.
        """
        if not cmd.raaga:
            return True
        if self.raagas.get(cmd.raaga) is None:
            self.status(f"I do not know a raaga called {cmd.raaga}.")
            return False
        if cmd.raaga.casefold() == (self.project.raaga.selected or "").casefold():
            return True
        try:
            self.select_raaga(cmd.raaga)
        except LockedContentError:
            # Silently composing in the locked raaga would answer a question
            # they did not ask; silently overriding the lock would undo a
            # decision they did.
            self.status(f"The raaga is locked to {self.project.raaga.selected}, "
                        f"so I did not switch to {cmd.raaga}. Unlock it first.")
            return False
        return True

    def make_variation(self, strength: float = 0.5) -> None:
        self.take_the_floor("tune variation")
        melody = self.project.melody()
        if melody is None:
            return self.generate_tune()
        raaga = self.composing_raaga()
        opts = self.melody_options()
        version = max(m.version for m in self.project.melodies) + 1
        self.status("Writing a variation...")

        def work(ctx: JobContext) -> MelodyVersion:
            ctx.progress(0.4, "Varying phrases")
            fresh = melody_engine.variation(melody, raaga, opts, version, strength)
            fresh.validation = validate(fresh, raaga).issues
            return fresh

        variation_ticket = self.song_work_ticket()
        self.jobs.submit("tune.variation", "melody:all", work,
                         on_done=lambda m: self._tune_ready(m, "Variation",
                                                            variation_ticket),
                         on_error=lambda e: self.error("tune", f"Variation failed: {e}"),
                         description="Tune variation")

    def regenerate_tune_section(self, section_id: str) -> None:
        self.take_the_floor("regenerate a section")
        melody = self.project.melody()
        if melody is None:
            return
        section = melody.section_by_id(section_id)
        if section is None:
            return
        assert_melody_editable(self.project, section.start, section.end)
        raaga = self.composing_raaga()
        opts = self.melody_options()
        version = max(m.version for m in self.project.melodies) + 1
        self.status(f"Rewriting {section.name}...")

        def work(ctx: JobContext) -> MelodyVersion:
            ctx.progress(0.5, f"Rewriting {section.name}")
            fresh = melody_engine.regenerate_section(melody, raaga, section_id, opts,
                                                    version)
            fresh.validation = validate(fresh, raaga).issues
            return fresh

        # A rewrite of one section depends on the tune it is rewriting and
        # on that section still being unlocked when it lands.
        section_ticket = self.song_work_ticket([section_id])
        self.jobs.submit("tune.section", f"melody:{section_id}", work,
                         on_done=lambda m: self._tune_ready(
                             m, f"Rewrote {section.name}", section_ticket),
                         on_error=lambda e: self.error("tune", f"Section rewrite failed: {e}"),
                         description=f"Rewrite {section.name}")

    def set_tempo(self, bpm: int) -> None:
        """Set the tempo, and remember that the creator chose it.

        Recording the preference is what makes the choice survive.  It used
        to be written only when there was no tune yet: change the tempo on
        an existing tune and the new speed lived in that melody alone, so
        the next Generate Tune - which starts from the brief - had nothing
        to tell it the creator had asked for anything.
        """
        self.take_the_floor("change the tempo")
        self.project.brief.tempo_preference = int(bpm)
        melody = self.project.melody()
        if melody is None:
            self._changed("tune.tempo", f"Tempo preference {bpm} bpm")
            return
        version = max(m.version for m in self.project.melodies) + 1
        fresh = melody_engine.retempo(melody, int(bpm), version)
        self._tune_ready(fresh, f"Tempo {bpm} bpm")

    def _tune_ready(self, melody: MelodyVersion, what: str,
                    ticket: Optional[Dict] = None) -> None:
        stale = self.stale_reason(ticket) if ticket else ""
        if stale:
            self.status(f"I did not keep that tune: {stale}.")
            return
        self.project.melodies.append(melody)
        self.project.approved_melody = melody.version
        self.project.current_stage = Stage.TUNE
        self._changed("tune.version", f"{what} tune v{melody.version}")

        # The agent marks its own work and remembers how it went.
        critique = ""
        try:
            _, evaluation = self.agent.record_composition(
                project_id=self.project.project_id, title=self.project.title,
                raaga=melody.raaga, brief=self.project.brief,
                notes=melody.notes, tempo_bpm=melody.tempo_bpm,
                structure={"sections": [s.name for s in melody.sections],
                           "version": melody.version},
                seed=melody.seed)
            self.last_evaluation = evaluation
            if evaluation.scores:
                critique = f" - the agent scores it {evaluation.overall():.2f}"
                if evaluation.recommendation:
                    critique += f"; {evaluation.recommendation}"
        except Exception as exc:  # noqa: BLE001 - critique must never block a tune
            log.warning("the agent could not mark the tune: %s", exc)

        shape = ""
        if getattr(melody, "plan_notes", None):
            # Said here as well as in the report, because the status is
            # what a creator reads when the tune arrives.
            shape = f" {melody.plan_notes[0]}"
        self.status(f"{what} tune v{melody.version} "
                    f"({melody.duration:.0f}s, {len(melody.notes)} notes)"
                    f"{critique}.{shape}")
        self.render(kind="tune", autoplay=False)

    def accept_tune(self, lock: bool = True) -> None:
        melody = self.project.melody()
        if melody is None:
            return
        melody.state = ApprovalState.LOCKED if lock else ApprovalState.APPROVED
        self.project.approved_melody = melody.version
        self.project.current_stage = Stage.LYRICS
        self._changed("tune.accept",
                      f"{'Locked' if lock else 'Approved'} tune v{melody.version}")

    def set_section_lock(self, section_id: str, locked: bool) -> None:
        melody = self.project.melody()
        if melody is None:
            return
        section = melody.section_by_id(section_id)
        if section is None:
            return
        section.locked = locked
        self._changed("tune.section_lock",
                      f"{'Locked' if locked else 'Unlocked'} {section.name}")

    def select_melody_version(self, version: int) -> None:
        if self.project.melody(version) is None:
            return
        self.project.approved_melody = version
        self._changed("tune.select", f"Switched to tune v{version}")
        self.render(kind="tune", autoplay=False)

    # ==================================================================
    # the learning agent
    # ==================================================================
    def agent_status(self) -> Dict[str, object]:
        return self.agent.status()

    def agent_events(self, limit: int = 30) -> List[Dict[str, object]]:
        return self.agent.recent_events(limit)

    def start_learning(self) -> bool:
        started = self.agent.start_learning()
        self.status("The agent is studying in the background."
                    if started else "The agent could not start learning.")
        return started

    def pause_learning(self) -> None:
        self.agent.pause_learning()
        self.status("Learning paused.")

    def resume_learning(self) -> None:
        self.agent.resume_learning()
        self.status("Learning resumed.")

    def stop_learning(self) -> None:
        self.agent.stop_learning(wait=False)
        self.status("Learning stopped.")

    def learn_now(self, cycles: int = 1) -> str:
        """Run the learning loop on this thread; used by the UI's step button."""
        steps = self.agent.learn(cycles)
        last = steps[-1] if steps else None
        message = last.summary() if last else "nothing to study"
        self.status(message)
        if self.on_project_changed:
            self.on_project_changed()
        return message

    def study_raaga(self, name: str) -> str:
        message = self.agent.study_raaga(name)
        self.status(message)
        return message

    #: Words that mean the question is about the tune on the desk rather
    #: than about a raaga.  "why" and "phrase" are not among them: they are
    #: the natural words for asking why a raaga was suggested or what
    #: phrases were learned, and routing on them sent every such question to
    #: the tune explainer as soon as any tune existed.
    #: "this phrase" and "you chose" belong here: they point at the line on
    #: the desk.  "what phrases has Keeravani learned" does not - it names a
    #: raaga and asks about records.  The discriminator is a demonstrative
    #: or a reference to the act of composing, not the word "phrase" itself.
    _ABOUT_THE_TUNE = ("this tune", "the tune", "this melody", "the melody",
                       "this line", "what you wrote", "what it wrote",
                       "this composition", "the composition",
                       "this phrase", "that phrase", "these phrases",
                       "you choose", "you chose", "did you choose",
                       "you pick", "you picked")

    def ask_agent(self, question: str) -> str:
        """Answer a question about the music or about what has been learned.

        Read-only: nothing here composes, trains, or changes the song.
        """
        low = (question or "").lower()
        melody = self.project.melody()
        named = self.raagas.find_in_text(question or "")
        subject = named.name if named else self.question_subject()

        about_the_tune = any(p in low for p in self._ABOUT_THE_TUNE)
        if melody is not None and about_the_tune:
            return self.agent.explain_choice(melody, subject or melody.raaga)

        if named:
            # Asking about a raaga by name makes it the subject of whatever
            # is asked next, so "what did it learn?" does not lose it.
            self._question_subject = named.name
        answer = self.knowledge_answer(subject, low)
        if answer:
            return answer
        return self.agent.explain(question, subject)

    def question_subject(self) -> str:
        """Whichever raaga the conversation is currently about.

        A follow-up rarely repeats the name.  Before this, every question
        re-derived the subject from its own words, so "has Keeravani been
        trained?" followed by "what did it learn?" answered the second
        about whatever the curriculum happened to be studying.
        """
        return (getattr(self, "_question_subject", "")
                or self.project.raaga.selected
                or (self.agent.curriculum.current_raaga() if self.agent else ""))

    def knowledge_answer(self, raaga: str, low: str) -> str:
        """Answer "has it been trained" and its follow-ups from real records.

        Returns "" when the question is not one of these, so the caller can
        fall through to the agent's own explanations.
        """
        if not raaga or self.agent is None:
            return ""
        wants_training = any(w in low for w in
                             ("trained", "training", "studied", "learned from",
                              "learnt from", "know about", "knowledge of"))
        wants_sources = any(w in low for w in
                            ("recording", "source", "where did", "which file",
                             "what did you hear", "evidence"))
        wants_gaps = any(w in low for w in
                         ("missing", "not know", "unknown", "gap", "lack"))
        # "what did it learn" is the second question in the sequence Krish
        # asked for, and it used to fall through to the curriculum branch,
        # which answered with a study stage rather than with anything the
        # agent had actually learned.
        wants_content = any(w in low for w in
                            ("what did it learn", "what has it learned",
                             "what did you learn", "what have you learned",
                             "what do you know", "what does it know"))
        if not (wants_training or wants_sources or wants_gaps or wants_content):
            return ""

        repo = self.agent.repo
        facts = repo.facts(raaga)
        ev = self._raaga_evidence(raaga, facts)
        # Only what is displayed is fetched.  Every number below comes from a
        # count, because an earlier version tallied a page of rows and then
        # reported the page size as the total: twenty-one recordings were
        # answered as twenty, and one was silently left out of the list.
        heard_phrases = repo.phrases(raaga=raaga, limit=4,
                                     origins=self._HEARD_ORIGINS)
        reference_phrases = repo.phrases(raaga=raaga, limit=4,
                                         origins=(provenance.REFERENCE,))

        if wants_sources:
            listed = repo.sources(raaga=raaga, limit=20)
            if not listed:
                return (f"Nothing has been ingested for {raaga}, so there are "
                        f"no recordings or references behind what I have.")
            # One heading cannot summarise a mixed list, and every version
            # of it inferred a universal outcome from part of the list.  Each
            # row carries its own status and whether anything was kept, so
            # the heading has nothing left to guess at.
            rows = [f"Sources and references on file for {raaga}:"]
            for s_ in listed:
                # What the source *is*, not what has been learned from it:
                # calling a queued recording "learned from a person's
                # recording" states the outcome of an analysis that has not
                # run, directly under a heading saying it has not run.
                kept = "kept" if s_.id in ev["kept_ids"] else "nothing kept"
                rows.append(f"  {s_.title[:60]} - "
                            f"{provenance.describe_source(s_.origin)}, "
                            f"{s_.status}, {kept}")
            total = ev["sources_total"]
            if total > len(listed):
                rows.append(f"  (the {len(listed)} most recent of {total} - "
                            f"a listing limit, not the total)")
            return "\n".join(rows)

        if wants_gaps:
            missing = []
            if not ev["from_recordings"]:
                missing.append("no phrases heard from a recording")
            if not facts:
                missing.append("no facts recorded")
            if not ev["sources_total"]:
                missing.append("no source ingested")
            entry = self.raagas.get(raaga)
            if entry is not None and entry.scale_only:
                missing.append("the library holds its scale only - no "
                               "characteristic phrases, resting notes or gamaka")
            if missing:
                answer = (f"For {raaga}, what I do not have: "
                          + "; ".join(missing) + ".")
                note = self._availability_note(ev)
                return f"{answer} {note}" if note else answer
            # Corroboration is about whether recordings agree with each other,
            # which is not something a source count can settle.  Saying "a
            # second recording would help" while holding twenty-one of them
            # was a sentence written for one case and printed for all of them.
            heard = ev["heard_sources"]
            if heard >= 2:
                corroboration = (f"Whether those {heard} recordings agree with "
                                 f"each other is something I have not "
                                 f"assessed, so I still cannot tell habit from "
                                 f"accident.")
            elif heard == 1:
                corroboration = ("What is missing is corroboration: a second "
                                 "recording would let me tell habit from "
                                 "accident.")
            else:
                corroboration = ("Nothing has been heard from a recording, so "
                                 "there is nothing yet to corroborate.")
            answer = (f"For {raaga} I have {ev['learned_total']} learned "
                      f"phrase(s), {len(facts)} fact(s) and "
                      f"{ev['sources_total']} source(s). {corroboration}")
            note = self._availability_note(ev)
            return f"{answer} {note}" if note else answer

        if wants_content:
            if not ev["learned_total"] and not facts:
                return (f"Nothing yet for {raaga} beyond the library's own "
                        f"reference.")
            # A fresh installation seeds fifteen structural facts per raaga
            # from the shipped library.  Listing those under "what I have
            # learned" says the same thing the training answer refuses to
            # say two lines above it, so the heading follows the same rule:
            # learning is what was retained from something studied.
            kinds = ev["facts_by_kind"]
            only_library = (kinds["reference"]
                            and not kinds["generated"]
                            and not kinds["unattributed"])
            if ev["learned_total"] or ev["learned_facts"]:
                rows = [f"What I have learned about {raaga}:"]
            elif only_library:
                rows = [f"Nothing has been learned from a recording for "
                        f"{raaga} yet. What the library ships with:"]
            else:
                rows = [f"Nothing has been learned from a recording for "
                        f"{raaga} yet. What is on file:"]
            # Each row says where it came from.  A heading cannot: a store
            # holding one library fact, one this system wrote and one with
            # no source at all had all three listed under whichever heading
            # the first of them earned, and once a single human phrase
            # arrives the heading becomes "what I have learned" and the
            # other two inherit a claim that was never made about them.
            for f in facts[:6]:
                rows.append(f"  {f.key}: {f.value} (confidence "
                            f"{f.confidence:.2f}) - "
                            f"{self._fact_source(f, ev)}")
            if len(facts) > 6:
                rows.append(f"  (6 of {len(facts)} fact(s) shown - a display "
                            f"limit, not the total)")
            if ev["from_recordings"]:
                rows.append(f"  and {ev['from_recordings']} phrase(s) heard in "
                            f"real recordings, the most trusted being:")
                for p in heard_phrases:
                    rows.append(f"    {' '.join(p.swaras)}  "
                                f"(confidence {p.confidence:.2f})")
            # The reference pack is the library's own material rendered so it
            # can be practised against.  It is legitimate to learn from and it
            # is not a performance, and calling it one - "heard in real
            # recordings" - was the answer claiming an ear it does not have.
            if ev["from_reference"]:
                rows.append(f"  and {ev['from_reference']} phrase(s) practised "
                            f"from the reference pack, which is the library's "
                            f"own material and not a performance:")
                for p in reference_phrases:
                    rows.append(f"    {' '.join(p.swaras)}  "
                                f"(confidence {p.confidence:.2f})")
            return "\n".join(rows)

        # wants_training.  What decides is what has been retained, never a
        # source's latest status.  A source is marked "analysed" when the
        # attempt finished, whether or not anything came of it; and a source
        # whose findings were kept can be re-run later and end at "failed".
        # Reading status as though it meant learning made the application
        # claim training from a recording it had kept nothing from, and deny
        # training it had genuinely done.
        learned_facts = ev["learned_facts"]
        unlearned = self._unlearned_facts(ev)
        if not ev["learned_total"] and not learned_facts:
            other = f" I have {unlearned}, which is not the same as having "\
                    f"heard it." if unlearned else ""
            note = self._availability_note(ev)
            return (f"No - {raaga} has had no training.{other}"
                    + (f" {note} Having a recording is not the same as having "
                       f"heard it." if note else ""))
        parts = [f"Yes - {raaga} has been trained."]
        held = []
        if ev["from_recordings"]:
            held.append(f"{ev['from_recordings']} phrase(s) heard in "
                        f"{ev['heard_sources']} recording(s)")
        if learned_facts:
            held.append(f"{learned_facts} fact(s) learned from a recording")
        if ev["from_reference"]:
            held.append(f"{ev['from_reference']} phrase(s) practised from the "
                        f"reference pack, which is the library's own material "
                        f"and not a performance")
        parts.append("  " + ", ".join(held) + ".")
        if not ev["from_recordings"] and not learned_facts:
            parts.append("  Nothing heard from a recording yet.")
        if unlearned:
            parts.append(f"  Alongside {unlearned}, which is not the same as "
                         f"having heard it.")
        if ev["by_origin"]:
            described = ", ".join(f"{n} {provenance.describe(o)}"
                                  for o, n in sorted(ev["by_origin"].items()))
            parts.append(f"  Where they came from: {described}.")
        note = self._availability_note(ev)
        if note:
            parts.append(f"  {note}")
        return "\n".join(parts)

    #: Source statuses, as research.py writes them.  None of them means
    #: "something was learned" - see ``_raaga_evidence``.
    _SOURCE_ANALYSED = ("analysed",)
    _SOURCE_WAITING = ("pending", "queued")
    _SOURCE_FAILED = ("failed",)
    _SOURCE_FOUND_NOTHING = ("empty",)
    #: Learnable origins that mean somebody performed something, as against
    #: the shipped library rendered for practice.
    _HEARD_ORIGINS = tuple(o for o in provenance.LEARNED_FROM
                           if o != provenance.REFERENCE)

    def _raaga_evidence(self, raaga: str, facts: Sequence = ()) -> Dict:
        """What is actually retained about a raaga, counted rather than sampled.

        Every answer above reads this one view, so "has it been trained",
        "what did it learn", "which recordings" and "what is missing" cannot
        disagree with each other about the same database.

        The organising rule is that evidence is what was kept, and a source's
        status is only the history of the last attempt on it.  The two are
        reported side by side and neither is inferred from the other.
        """
        repo = self.agent.repo
        by_origin = repo.count_phrases_by_origin(raaga, learned_only=True)
        index = repo.source_index(raaga)
        origin_of = {sid: origin for sid, _status, origin in index}

        # Sources that actually yielded something still held: a phrase, or a
        # fact that names them.  A fact whose source is the shipped library
        # is the reference book, not a recording.
        heard_ids = set(repo.phrase_source_ids(raaga,
                                               origins=self._HEARD_ORIGINS))
        kept_ids = set(repo.phrase_source_ids(
            raaga, origins=tuple(provenance.LEARNED_FROM)))
        # Every fact is classified by the source it names.  Counting one
        # bucket and calling the remainder the shipped library invented a
        # provenance for anything that had none - which is the mistake this
        # whole sequence has been about, arrived at by subtraction.
        by_kind = {"learned": 0, "reference": 0, "generated": 0,
                   "unattributed": 0}
        for f in facts:
            source_id = getattr(f, "source_id", "")
            origin = origin_of.get(source_id, "") if source_id else ""
            if origin in self._HEARD_ORIGINS:
                by_kind["learned"] += 1
                kept_ids.add(source_id)
                heard_ids.add(source_id)
            elif origin == provenance.REFERENCE:
                by_kind["reference"] += 1
                kept_ids.add(source_id)
            elif origin == provenance.GENERATED:
                by_kind["generated"] += 1
            else:
                # No source id, a source that is not on file, or one whose
                # own origin was never recorded.  Saying anything more than
                # that would be making it up.
                by_kind["unattributed"] += 1
        learned_facts = by_kind["learned"]

        def by_status(statuses: Sequence[str], reference: bool) -> List[str]:
            return [sid for sid, status, origin in index
                    if status in statuses
                    and provenance.may_be_learned_from(origin)
                    and (origin == provenance.REFERENCE) is reference]

        analysed = by_status(self._SOURCE_ANALYSED, False)
        from_reference = by_origin.get(provenance.REFERENCE, 0)
        return {
            "by_origin": by_origin,
            "learned_total": sum(by_origin.values()),
            "from_recordings": sum(by_origin.values()) - from_reference,
            "from_reference": from_reference,
            "learned_facts": learned_facts,
            "facts_by_kind": by_kind,
            "source_origins": origin_of,
            "heard_ids": heard_ids,
            "kept_ids": kept_ids,
            "heard_sources": len(heard_ids),
            "analysed_without_result": len([s for s in analysed
                                            if s not in kept_ids]),
            "waiting": len(by_status(self._SOURCE_WAITING, False)),
            "failed": len(by_status(self._SOURCE_FAILED, False)),
            "found_nothing": len(by_status(self._SOURCE_FOUND_NOTHING, False)),
            "reference_packs": len(by_status(self._SOURCE_ANALYSED, True)),
            "sources_total": len(index),
        }

    @staticmethod
    def _fact_source(fact, ev: Dict) -> str:
        """Where one fact came from, by the source it names.

        The same classification the totals use, applied to the row the
        creator is actually reading.
        """
        source_id = getattr(fact, "source_id", "")
        origin = ev["source_origins"].get(source_id, "") if source_id else ""
        if not origin or origin == provenance.UNKNOWN:
            return "no identified source"
        return provenance.describe(origin)

    @staticmethod
    def _unlearned_facts(ev: Dict) -> str:
        """Facts on file that are not learning, each named by what it is.

        The library's reference book, this system's own output and a fact
        whose source was never recorded are three different things, and only
        the first of them is knowledge the application was given.
        """
        kinds = ev["facts_by_kind"]
        bits = []
        if kinds["reference"]:
            bits.append(f"the library's built-in reference "
                        f"({kinds['reference']} fact(s))")
        if kinds["generated"]:
            bits.append(f"{kinds['generated']} fact(s) this system wrote "
                        f"itself")
        if kinds["unattributed"]:
            bits.append(f"{kinds['unattributed']} fact(s) on file with no "
                        f"identified source")
        return ", ".join(bits)

    @staticmethod
    def _availability_note(ev: Dict) -> str:
        """Sources on hand that have not become learning.

        Named separately because having a recording and having heard one are
        different events, and reporting the first as the second is how the
        application came to say a raaga had been trained from a recording it
        had never opened.  This is the history of the attempts; it never
        contradicts what was kept, because it never speaks about it.
        """
        bits = []
        if ev["waiting"]:
            bits.append(f"{ev['waiting']} waiting to be analysed")
        if ev["failed"]:
            bits.append(f"{ev['failed']} whose most recent analysis failed")
        if ev["found_nothing"]:
            bits.append(f"{ev['found_nothing']} analysed without yielding a "
                        f"phrase")
        if ev["analysed_without_result"]:
            bits.append(f"{ev['analysed_without_result']} analysed with "
                        f"nothing kept from it")
        if not bits:
            return ""
        return "Latest analysis status: " + ", ".join(bits) + "."

    def agent_knowledge(self, name: str = "") -> str:
        return self.agent.knowledge_report(
            name or self.project.raaga.selected
            or self.agent.curriculum.current_raaga())

    def critique_tune(self) -> str:
        """What the agent thinks of the tune on the desk."""
        melody = self.project.melody()
        if melody is None:
            return "There is no tune to look at yet."
        raaga = self.composing_raaga()
        evaluation = self.agent.evaluator(raaga.name).evaluate(
            melody.notes, raaga, tonic_midi=melody.tonic_midi,
            brief=self.project.brief, tempo_bpm=melody.tempo_bpm,
            expected_seconds=self.project.brief.duration_target,
            learned_phrases=self.agent.phrase_bank(raaga.name))
        self.last_evaluation = evaluation
        try:
            self.agent.record_lessons(
                evaluation, raaga=raaga.name, task="composition",
                method="critique", result=evaluation.overall(),
                source_run=self.project.project_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record lessons from this critique: %s", exc)
        return evaluation.report()

    def give_feedback(self, text: str) -> str:
        """Creator feedback is education: it is stored and it changes behaviour."""
        melody = self.project.melody()
        swaras = [n.swara for n in melody.notes] if melody else None
        answer = self.agent.record_feedback(
            text, raaga=self.project.raaga.selected, swaras=swaras,
            target_kind="composition", target_id=self.project.project_id)
        self.project.log_history("agent.feedback", text[:200])
        self._changed("", "", undoable=False)
        self.status(answer)
        try:
            self._record_feedback_lessons(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record a lesson from this feedback: %s", exc)
        try:
            sentiment = self.agent.feedback_sentiment(text)
            if sentiment in ("positive", "negative"):
                raaga = self.project.raaga.selected or self.agent.curriculum.current_raaga()
                self.agent.record_field_feedback(
                    raaga, text, sentiment == "positive", self.last_evaluation,
                    self.project.project_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record field feedback for the factory: %s", exc)
        return answer

    def _record_feedback_lessons(self, text: str) -> None:
        """Negative feedback is high-weight evidence (spec section 26)."""
        if self.agent.feedback_sentiment(text) != "negative":
            return
        raaga = self.project.raaga.selected or self.agent.curriculum.current_raaga()
        self.agent.repo.add_lesson(Lesson(
            raaga=raaga, kind="creator_feedback", dimension="creator",
            failure_reason=text[:200], task="composition",
            method="creator feedback", confidence=0.95,
            source_run=self.project.project_id))
        # The creator's words in the evaluator's vocabulary, so the next
        # tune's guidance acts on them even when the critic found nothing.
        for kind in self.agent.feedback_kinds(text):
            self.agent.repo.add_lesson(Lesson(
                raaga=raaga, kind=kind, dimension="creator",
                failure_reason=f"the creator said: {text[:160]}",
                correction=kind.replace("_", " "), task="composition",
                method="creator feedback", confidence=0.95,
                source_run=self.project.project_id))
        if self.last_evaluation is not None:
            self.agent.record_lessons(
                self.last_evaluation, raaga=raaga, task="composition",
                method="creator feedback", result=self.last_evaluation.overall(),
                source_run=self.project.project_id, confidence=0.95)

    def validation_report(self) -> str:
        melody = self.project.melody()
        raaga = self.current_raaga()
        if melody is None or raaga is None:
            return "No tune yet."
        # The planner's reasons go first.  They were being stored and never
        # shown: this recomputed the raaga check and returned only that, so
        # a song that could not be the length it was asked to be said
        # "no issues found" and explained nothing.
        report = validate(melody, raaga).summary()
        notes = list(getattr(melody, "plan_notes", []) or [])
        if notes:
            return "\n".join(notes) + "\n\n" + report
        return report

    # ==================================================================
    # lyrics
    # ==================================================================
    def generate_lyrics(self, seed: Optional[int] = None,
                        section_ids: Optional[Sequence[str]] = None) -> None:
        """Write words for the tune, or for chosen sections of it.

        Krish's path is to settle the Pallavi before anything else exists:
        select that section, write words fitted to the tune it already has,
        and leave every other section - and every locked line - as it was.
        Passing no selection writes for the whole song, as before.
        """
        self.take_the_floor("write lyrics")
        melody = self.project.melody()
        if melody is None:
            self.status("Write a tune first - the lyrics are fitted to it.")
            return
        chosen = self._singable_sections(melody, section_ids)
        if chosen is None:
            return
        brief = self.project.brief
        version = max((l.version for l in self.project.lyrics), default=0) + 1
        previous = self.project.lyrics_version()
        llm = self.providers.llm
        ticket = self.song_work_ticket(chosen)
        self.status("Writing lyrics to fit the tune...")

        def work(ctx: JobContext) -> LyricsVersion:
            ctx.progress(0.3, "Measuring phrases")
            return lyric_generator.generate(
                melody, brief, version=version,
                seed=seed if seed is not None else version * 17, llm=llm,
                previous=previous, section_ids=chosen)

        def done(lyrics: LyricsVersion) -> None:
            stale = self.stale_reason(ticket)
            if stale:
                self.status(f"I did not keep those lyrics: {stale}.")
                return
            lyrics = self._merge_sections(lyrics, chosen)
            # Recomputed here, not at submission: another version may have
            # arrived while this one was being written.
            lyrics.version = max((l.version for l in self.project.lyrics),
                                 default=0) + 1
            self.project.lyrics.append(lyrics)
            self.project.approved_lyrics = lyrics.version
            self.project.current_stage = Stage.VOICE
            self._changed("lyrics.version", f"Lyrics v{lyrics.version}")
            where = ""
            if chosen:
                names = [s.name for s in melody.sections if s.id in set(chosen)]
                where = f" for {', '.join(names)}" if names else ""
            self.status(f"Lyrics v{lyrics.version}: {len(lyrics.lines)} lines "
                        f"fitted{where}")

        self.jobs.submit("lyrics.generate", "lyrics", work, on_done=done,
                         on_error=lambda e: self.error("lyrics", f"Lyrics failed: {e}"),
                         description="Write lyrics for the tune")

    def _singable_sections(self, melody, section_ids
                           ) -> Optional[Tuple[str, ...]]:
        """Settle which sections a request is about.  None means refuse.

        An empty tuple means the whole song, which is what no selection has
        always meant.  A selection that names an instrumental section, or a
        section this tune does not have, is answered rather than quietly
        widened - silently writing for the whole song is the one outcome
        choosing a section was meant to prevent.
        """
        if not section_ids:
            return ()
        known = {s.id: s for s in melody.sections}
        unknown = [sid for sid in section_ids if sid not in known]
        if unknown:
            self.status("That section is not part of this tune.")
            return None
        singable = tuple(sid for sid in section_ids
                         if not known[sid].kind.instrumental)
        if not singable:
            names = ", ".join(known[sid].name for sid in section_ids)
            self.status(f"{names} is instrumental - it has music but no "
                        f"words to sing.")
            return None
        skipped = [known[sid].name for sid in section_ids
                   if known[sid].kind.instrumental]
        if skipped:
            self.status(f"Leaving {', '.join(skipped)} instrumental.")
        locked = [known[sid].name for sid in singable if known[sid].locked]
        if locked:
            self.status(f"{', '.join(locked)} is locked. Unlock it first.")
            return None
        return singable

    def edit_lyric_line(self, line_id: str, text: str) -> List[str]:
        lyrics = self.project.lyrics_version()
        melody = self.project.melody()
        if lyrics is None or melody is None:
            return []
        warnings = lyric_fitting.refit_line(lyrics, melody, line_id, text)
        self._changed("lyrics.edit", "Edited a lyric line")
        return warnings

    def regenerate_lyric_line(self, line_id: str) -> List[str]:
        lyrics = self.project.lyrics_version()
        melody = self.project.melody()
        if lyrics is None or melody is None:
            return []
        warnings = lyric_generator.regenerate_line(
            lyrics, melody, line_id, self.project.brief, llm=self.providers.llm)
        self._changed("lyrics.line", "Rewrote a lyric line")
        return warnings

    def set_lyric_line_lock(self, line_id: str, locked: bool) -> None:
        lyrics = self.project.lyrics_version()
        if lyrics is None:
            return
        line = lyrics.line_by_id(line_id)
        if line is None:
            return
        line.locked = locked
        self._changed("lyrics.lock", f"{'Locked' if locked else 'Unlocked'} a line")

    def accept_lyrics(self) -> None:
        lyrics = self.project.lyrics_version()
        if lyrics is None:
            return
        lyrics.state = ApprovalState.LOCKED
        self.project.approved_lyrics = lyrics.version
        self.project.current_stage = Stage.VOICE
        self._changed("lyrics.accept", f"Accepted lyrics v{lyrics.version}")

    def lyric_alignment(self) -> str:
        lyrics = self.project.lyrics_version()
        melody = self.project.melody()
        if lyrics is None or melody is None:
            return "No lyrics yet."
        return lyric_fitting.alignment_report(lyrics, melody)

    # ==================================================================
    # voice
    # ==================================================================
    def current_voice(self) -> VoiceProfile:
        profile = self.voices.get(self.project.voice_profile_id)
        if profile is None:
            profile = self.voices.default()
            self.project.voice_profile_id = profile.id
        return profile

    def set_voice(self, profile_id: str) -> VoiceProfile:
        profile = self.voices.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        self.project.voice_profile_id = profile.id
        self._changed("voice.set", f"Singer: {profile.name}")
        return profile

    def set_vocal_direction(self, **fields) -> None:
        direction = self.project.vocal_direction
        for key, value in fields.items():
            if hasattr(direction, key):
                setattr(direction, key, value)
        self._changed("voice.direction",
                      f"Vocal direction: {direction.style} "
                      f"(intensity {direction.intensity:.2f})")

    def render_vocal(self, kind: str = "preview", autoplay: bool = False,
                     section_ids: Optional[Sequence[str]] = None,
                     then_play: Optional[Tuple[float, float, bool]] = None
                     ) -> None:
        """kind is 'preview' or 'master' (the studio vocal-only version).

        Rendering finishes quietly.  It used to start the player itself,
        which meant a creator who asked for a studio master got sound in
        the room without asking for it, and an unattended test made noise.
        Play Vocal is the action that plays; this one is the action that
        prepares something to play.

        ``section_ids`` sings only those sections, at their place in the
        song and over its full length, so the take still lines up with the
        arrangement and the two can be heard together.

        ``then_play`` is a section preview's follow-on: mix the song and
        play that span, once this take is in.  It travels in this job's
        closure rather than on the controller.  It used to be a field, and
        a field is reachable by every other render - so a preview whose
        vocal failed, or whose song was replaced, left it lying there for
        the next successful take to pick up and play.
        """
        self.take_the_floor("render the vocal")
        melody = self.project.melody()
        if melody is None:
            self.status("There is no tune to sing yet.")
            return
        chosen = self._singable_sections(melody, section_ids)
        if chosen is None:
            return
        lyrics = self.project.lyrics_version()
        profile = self.current_voice()
        direction = self.project.vocal_direction
        sr = self.sample_rate
        provider = self.providers.voice
        version = len(self.project.vocal_renders) + 1
        self.status("Rendering the vocal..." if kind == "preview"
                    else "Producing the studio vocal-only master...")

        ticket = self.song_work_ticket(chosen)
        ticket["lyric_fingerprint"] = self.lyric_fingerprint(lyrics)
        ticket["voice_profile_id"] = profile.id
        self._vocal_request += 1
        request_id = self._vocal_request

        def work(ctx: JobContext) -> Tuple[VocalRender, np.ndarray]:
            ctx.progress(0.2, "Singing the line")
            raw = provider.render_vocal(melody, lyrics, profile, direction, sr,
                                        melody.duration + 1.0, seed=version * 7,
                                        section_ids=chosen or None)
            ctx.progress(0.6, "Vocal production chain")
            if kind == "master":
                produced = mastering.master_vocal_only(raw, sr, direction)
            else:
                produced = mastering.quick_preview(raw, sr, direction)
            ctx.progress(0.95, "Writing the take")
            take = VocalRender(version=version, kind=kind,
                               melody_version=melody.version,
                               lyrics_version=lyrics.version if lyrics else 0,
                               voice_profile_id=profile.id, direction=direction,
                               duration=len(produced) / sr)
            return take, produced

        def done(result: Tuple[VocalRender, np.ndarray]) -> None:
            # Before the file is written, not after: a take belonging to a
            # song we have left would otherwise be saved into the new
            # song's renders directory and then played to the creator as
            # theirs.
            # Checked before the artifact is written and before anything
            # is played.  A worker can finish while its completion waits in
            # the queue for the interface timer, and in that window the
            # creator can ask for another render; the one they asked for
            # second is the one they are waiting to hear.
            if request_id != self._vocal_request:
                self.status("I did not keep that take: you asked for another "
                            "one while it was rendering.")
                return
            stale = self.stale_reason(ticket)
            if stale:
                self.status(f"I did not keep that take: {stale}.")
                return
            take, audio = result
            path = self._write_artifact(
                "renders", f"vocal_{kind}_v{take.version}.wav", audio)
            take.audio_path = str(path)
            self.project.vocal_renders.append(take)
            if kind == "master":
                self.project.vocal_master_id = take.id
            self.project.current_stage = Stage.ARRANGEMENT
            self._cache_render("vocal_master" if kind == "master" else "vocal_preview",
                               audio, str(path))
            self._changed("voice.render",
                          f"Vocal {kind} take v{take.version}")
            where = ""
            if chosen:
                names = [sec.name for sec in melody.sections
                         if sec.id in set(chosen)]
                where = f" ({', '.join(names)} only)" if names else ""
            self.status(f"Vocal {kind} ready{where} - "
                        f"{mastering.report(audio, self.sample_rate)}")
            if then_play is not None:
                start, end, play = then_play
                self.render(kind="full", autoplay=play,
                            play_range=(start, end))
            elif autoplay:
                self.play_render("vocal_master" if kind == "master"
                                 else "vocal_preview")

        self.jobs.submit(f"voice.{kind}", "vocal", work, on_done=done,
                         on_error=lambda e: self.error("voice", f"Vocal render failed: {e}"),
                         description=f"Render the {kind} vocal")

    def _span_name(self, melody, span) -> str:
        """What the creator would call the stretch they are hearing."""
        if melody is None or not span:
            return "the whole song"
        start, end = span
        for sec in melody.sections:
            if abs(sec.start - start) < 0.01 and abs(sec.end - end) < 0.01:
                return sec.name
        return f"{start:.1f}s to {end:.1f}s"

    def audition_scope(self) -> str:
        """The stretch a comparison would play, in the creator's words."""
        return self._audition[1] if self._audition else "the whole song"

    def preview_section(self, section_id: str, autoplay: bool = True) -> None:
        """Hear one section: its words sung, over its accompaniment.

        Krish asked to hear the words and the tune together with the
        instruments behind them, not a bare vocal - so this is not a second
        rendering path.  It sings the chosen section, then asks for the
        ordinary full mix and plays only that section's span of it, which
        is why the vocal take is rendered at full song length with silence
        around it: the two line up because they are the same timeline.
        """
        melody = self.project.melody()
        if melody is None:
            self.status("There is no tune yet.")
            return
        section = next((sec for sec in melody.sections
                        if sec.id == section_id), None)
        if section is None:
            self.status("That section is not part of this tune.")
            return
        if section.kind.instrumental:
            # Nothing is sung here, so there is nothing to wait for: the
            # accompaniment alone is the honest answer.
            self.status(f"{section.name} is instrumental - playing its "
                        f"accompaniment.")
            self.render(kind="full", autoplay=autoplay,
                        play_range=(section.start, section.end))
            return
        self.render_vocal(kind="preview", autoplay=False,
                          section_ids=[section_id],
                          then_play=(section.start, section.end, autoplay))

    def create_voice_from_recordings(self, paths: Sequence[str], name: str,
                                     gender: str = "") -> VoiceProfile:
        profile = self.voices.create_from_recording(list(paths), name, gender)
        self.project.voice_profile_id = profile.id
        self._changed("voice.profile", f"Created voice profile {profile.name}")
        return profile

    # ==================================================================
    # arrangement
    # ==================================================================
    def arrangement(self) -> Optional[ArrangementVersion]:
        return self.project.arrangement()

    def _ensure_arrangement(self) -> ArrangementVersion:
        arrangement = self.project.arrangement()
        if arrangement is None:
            arrangement = arranger.new_version(None)
            self.project.arrangements.append(arrangement)
            self.project.current_arrangement = arrangement.version
        return arrangement

    def add_instrument(self, instrument: str, start: float, end: float,
                       role: str = "", intensity: float = 0.6) -> None:
        melody = self.project.melody()
        if melody is None:
            self.status("Write a tune before arranging.")
            return
        raaga = self.require_raaga()
        inst = catalog.get(instrument)
        if inst is None:
            close = catalog.closest(instrument)
            names = ", ".join(c.name for c in close) or "nothing similar"
            self.error("arrangement",
                       f"'{instrument}' is not in the instrument catalog. "
                       f"Closest available: {names}.")
            return
        arrangement = self._ensure_arrangement()
        try:
            track, region = arranger.add_instrument(
                arrangement, melody, raaga, inst.key, start, end, role=role,
                intensity=intensity)
        except LockedContentError as exc:
            self.error("arrangement", str(exc))
            return
        self.project.current_stage = Stage.ARRANGEMENT
        self.context.last_track_id = track.id
        self.context.last_region_id = region.id
        self.context.last_instrument = inst.key
        self._changed("arrange.add",
                      f"Added {inst.name} {start:.0f}-{end:.0f}s ({track.role})")
        self.status(f"Added {inst.name} from {start:.0f}s to {end:.0f}s")
        self.render(kind="full", autoplay=False)

    def remove_instrument(self, instrument: str, start: Optional[float] = None,
                          end: Optional[float] = None) -> None:
        arrangement = self.project.arrangement()
        if arrangement is None:
            return
        inst = catalog.get(instrument)
        try:
            removed = arranger.remove_instrument(arrangement, instrument, start, end)
        except LockedContentError as exc:
            self.error("arrangement", str(exc))
            return
        if not removed:
            self.status(f"{inst.name if inst else instrument} was not playing there.")
            return
        self._changed("arrange.remove",
                      f"Removed {inst.name if inst else instrument}"
                      + (f" {start:.0f}-{end:.0f}s" if start is not None else ""))
        self.render(kind="full", autoplay=False)

    def replace_instrument(self, old: str, new: str, start: Optional[float] = None,
                           end: Optional[float] = None) -> None:
        melody = self.project.melody()
        arrangement = self.project.arrangement()
        if melody is None or arrangement is None:
            return
        new_inst = catalog.get(new)
        if new_inst is None:
            close = ", ".join(c.name for c in catalog.closest(new))
            self.error("arrangement",
                       f"'{new}' is not available. Closest: {close or 'nothing similar'}.")
            return
        raaga = self.require_raaga()
        try:
            arranger.replace_instrument(arrangement, melody, raaga, old, new_inst.key,
                                        start, end)
        except LookupError as exc:
            self.status(str(exc))
            return
        except LockedContentError as exc:
            self.error("arrangement", str(exc))
            return
        old_inst = catalog.get(old)
        self.context.last_instrument = new_inst.key
        self._changed("arrange.replace",
                      f"Replaced {old_inst.name if old_inst else old} with {new_inst.name}")
        self.status(f"Replaced {old_inst.name if old_inst else old} with {new_inst.name}")
        self.render(kind="full", autoplay=False)

    def suggest_instruments(self, feel_words: Sequence[str], role: str = "",
                            limit: int = 4) -> List[Tuple[catalog.Instrument, float]]:
        avoid = self.project.brief.instruments_avoided
        ranked = catalog.suggest_for_feel(feel_words, avoid, role=role, limit=limit)
        llm = self.providers.llm
        if llm is not None and llm.available and feel_words:
            try:
                keys = llm.suggest_instruments(", ".join(feel_words), catalog.keys())
                boosted = [(catalog.get(k), 10.0 - i) for i, k in enumerate(keys)
                           if catalog.get(k)]
                if boosted:
                    seen = {i.key for i, _ in boosted}
                    ranked = boosted + [(i, s) for i, s in ranked if i.key not in seen]
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM instrument advice failed: %s", exc)
        return ranked[:limit]

    def auto_arrange(self) -> None:
        self.take_the_floor("arrange")
        melody = self.project.melody()
        if melody is None:
            self.status("Write a tune before arranging.")
            return
        raaga = self.require_raaga()
        brief = self.project.brief
        previous = self.project.arrangement()
        # Cast on this thread, where the settings and the provider live, and
        # hand the arrangement the answer.  The audition uses the same call,
        # so the two cannot disagree about who is playing the melody.
        lead = self.cast_lead()
        beat = self.project.beat()
        self.status("Building a first arrangement...")

        def work(ctx: JobContext) -> ArrangementVersion:
            ctx.progress(0.3, f"Choosing instruments - {lead.describe()}")
            return arranger.auto_arrange(melody, raaga, brief, previous=previous,
                                         lead=lead.instrument, beat=beat)

        arrange_ticket = self.song_work_ticket()

        def done(arrangement: ArrangementVersion) -> None:
            stale = self.stale_reason(arrange_ticket)
            if stale:
                self.status(f"I did not keep that arrangement: {stale}.")
                return
            self.project.arrangements.append(arrangement)
            self.project.current_arrangement = arrangement.version
            self.project.current_stage = Stage.ARRANGEMENT
            names = ", ".join(t.label for t in arrangement.tracks)
            self._changed("arrange.auto", f"Auto arrangement v{arrangement.version}")
            self.status(f"Arrangement v{arrangement.version}: {names}")
            self.render(kind="full", autoplay=False)

        self.jobs.submit("arrange.auto", "arrangement:auto", work, on_done=done,
                         on_error=lambda e: self.error("arrangement",
                                                       f"Arrangement failed: {e}"),
                         description="Build the arrangement")

    def regenerate_region(self, track_id: str, region_id: str) -> None:
        melody = self.project.melody()
        arrangement = self.project.arrangement()
        if melody is None or arrangement is None:
            return
        try:
            arranger.regenerate_region(arrangement, melody, self.require_raaga(),
                                       track_id, region_id)
        except LockedContentError as exc:
            self.error("arrangement", str(exc))
            return
        self._changed("arrange.regenerate", "Regenerated a region")
        self.render(kind="full", autoplay=False)

    def set_mix(self, *, vocal_gain: Optional[float] = None,
                reverb: Optional[float] = None, room: Optional[float] = None,
                effects: Optional[bool] = None) -> None:
        """Change how the song is balanced.  Saved with it, and undoable.

        These are decisions about the song, not about one playback, which
        is why they live on the project and come back when it is reopened.
        """
        settings = self.project.mix_settings
        if vocal_gain is not None:
            settings.vocal_gain = max(0.0, min(2.0, float(vocal_gain)))
        if reverb is not None:
            settings.reverb = max(0.0, min(2.0, float(reverb)))
        if room is not None:
            settings.room = max(0.05, min(0.95, float(room)))
        if effects is not None:
            settings.effects = bool(effects)
        self._changed("mix.settings", f"Mix: {settings.describe()}")
        self.status(f"Mix: {settings.describe()}")

    def compare_dry(self) -> None:
        """Render the same material with the room taken away.

        The comparison only means something if nothing else changes, so
        this renders from the takes already in hand rather than singing or
        playing anything again.
        """
        if self.project.melody() is None:
            self.status("There is nothing to compare yet.")
            return
        was = self.project.mix_settings.effects
        self.project.mix_settings.effects = not was
        # Say what actually changes.  A studio vocal take carries the
        # processing it was mastered with, and this switch does not reach
        # inside it, so calling the result "dry" would be a promise the
        # render does not keep.
        state = ("without the instrument room" if was
                 else "with the instrument room")
        where = self.audition_scope()
        self._changed("mix.compare", f"Heard {where} {state}")
        self.render(kind="full", autoplay=True,
                    play_range=self._audition[0] if self._audition else None)

    def set_track_flag(self, track_id: str, *, mute: Optional[bool] = None,
                       solo: Optional[bool] = None, locked: Optional[bool] = None,
                       gain: Optional[float] = None,
                       pan: Optional[float] = None) -> None:
        arrangement = self.project.arrangement()
        if arrangement is None:
            return
        track = arrangement.track_by_id(track_id)
        if track is None:
            return
        if mute is not None:
            track.mute = mute
        if solo is not None:
            track.solo = solo
        if locked is not None:
            track.locked = locked
        if gain is not None:
            track.gain = max(0.0, min(2.0, gain))
        if pan is not None:
            track.pan = max(-1.0, min(1.0, pan))
        self._changed("arrange.track", f"Updated {track.label}")
        if mute is not None or solo is not None or gain is not None or pan is not None:
            self.render(kind="full", autoplay=False)

    def change_level(self, instrument: str, factor: float,
                     start: Optional[float] = None,
                     end: Optional[float] = None) -> None:
        arrangement = self.project.arrangement()
        if arrangement is None:
            return
        touched = []
        for track in arrangement.tracks:
            if instrument and track.instrument != instrument:
                continue
            if start is not None and end is not None:
                regions = track.regions_in(start, end)
                if not regions:
                    continue
                for r in regions:
                    if r.locked:
                        continue
                    r.gain = max(0.05, min(2.0, r.gain * factor))
                touched.append(track.label)
            else:
                track.gain = max(0.05, min(2.0, track.gain * factor))
                touched.append(track.label)
        if not touched:
            self.status("Nothing to change there.")
            return
        self._changed("arrange.level",
                      f"{'Raised' if factor > 1 else 'Lowered'} {', '.join(touched)}")
        self.render(kind="full", autoplay=False)

    def lock_range(self, start: float, end: float, locked: bool = True) -> None:
        arrangement = self.project.arrangement()
        melody = self.project.melody()
        n = 0
        if arrangement is not None:
            n = arranger.lock_range(arrangement, start, end, locked)
        if melody is not None:
            for section in melody.sections:
                if section.start < end and start < section.end:
                    section.locked = locked
        self._changed("region.lock",
                      f"{'Locked' if locked else 'Unlocked'} {start:.0f}-{end:.0f}s "
                      f"({n} region(s))")

    # ==================================================================
    # rendering, mixing, playback
    # ==================================================================
    def _write_artifact(self, category: str, name: str,
                        audio: np.ndarray) -> Path:
        if self.project_dir is None:
            self.save()
        path = ProjectStore.artifact_path(self.project_dir, category, name)
        export_engine.write_wav(path, audio, self.sample_rate)
        self.store.note_artifact(self.project_dir, path, category)
        return path

    def take_the_floor(self, what: str = "") -> bool:
        """A new creative action silences whatever is playing.

        Stop is for stopping (specification 12.1).  It was also the way to
        get out of the last thing before starting the next: press Generate
        while a tune is playing and the old audio kept going underneath the
        new work, so the habit became Stop-then-do, every time.

        The job manager already supersedes in-flight work for the same
        target, which is the other half of 12.3; this is the playback half.
        Returns whether anything was actually stopped, so a caller can say
        so if it matters.

        Not called by ``play_render`` - playing something *is* the action
        there, and ``load`` already stops before it swaps the audio.
        """
        if not self.playback.playing:
            return False
        self.playback.stop()
        log.debug("stopped playback for %s", what or "a new action")
        return True

    def _cache_render(self, kind: str, audio: np.ndarray, path: str = "") -> None:
        self._renders[kind] = RenderedAudio(kind=kind, audio=audio,
                                            sample_rate=self.sample_rate, path=path,
                                            duration=len(audio) / self.sample_rate)
        if self.on_render:
            self.on_render(kind)

    def _restore_renders(self) -> None:
        """Reload the audio referenced by a reopened project."""
        import soundfile as sf
        wanted = []
        take = self.project.vocal_master
        if take and take.audio_path:
            wanted.append(("vocal_master", take.audio_path))
        latest = self.project.latest_vocal
        if latest and latest.audio_path and latest.kind == "preview":
            wanted.append(("vocal_preview", latest.audio_path))
        for kind in ("full", "instrumental"):
            mix = self.project.latest_mix(kind)
            if mix and mix.audio_path:
                wanted.append((kind, mix.audio_path))
        melody = self.project.melody()
        if melody and melody.audio_path:
            wanted.append(("tune", melody.audio_path))
        for kind, path in wanted:
            try:
                if Path(path).exists():
                    audio, sr = sf.read(path, dtype="float32", always_2d=True)
                    self._renders[kind] = RenderedAudio(
                        kind=kind, audio=audio, sample_rate=sr, path=path,
                        duration=len(audio) / sr)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not reload %s: %s", path, exc)

    def render(self, kind: str = "full", autoplay: bool = False,
               play_range: Optional[Tuple[float, float]] = None) -> None:
        """Render one of the audio products in the background."""
        melody = self.project.melody()
        scope_name = self._span_name(melody, play_range)
        if melody is None:
            self.status("Nothing to render yet.")
            return
        sr = self.sample_rate
        total = max(melody.duration, self.project.material_duration) + 0.5
        arrangement = self.project.arrangement()
        # The same selection Play Vocal uses.  Fixing the button and
        # leaving the mix reading "master first, preview otherwise" meant
        # the creator could hear the newer take on its own and the older
        # one inside the song - the wrong voice or the wrong words, in the
        # thing they were actually judging.
        current = self.current_vocal_take()
        vocal_take = current[1] if current else self.project.latest_vocal
        vocal_audio = None
        if kind in ("full", "vocal_only") and current is not None:
            cached = self._renders.get(current[0])
            if cached is not None:
                vocal_audio = cached.audio
        raaga = self.require_raaga()
        provider = self.providers.music
        # Read once, here, so every part of one render is mixed to the same
        # settings even if the creator moves a control while it runs.
        settings = replace(self.project.mix_settings)
        self.status(f"Rendering the {kind.replace('_', ' ')}..."
                    if not play_range else f"Rendering {scope_name}...")

        def work(ctx: JobContext) -> Tuple[str, np.ndarray, dict]:
            if kind == "tune":
                # Hummed, not played (specification 10.1-10.3).  A tune was
                # previewed on an instrument, which asked the creator to
                # judge two things at once - the line and the timbre - and
                # every instrument here is additive synthesis, so a "violin"
                # is seven harmonics and an envelope.  It sounded like a
                # keyboard because that is what it is, and the melody was
                # blamed for the sound.
                #
                # A singer with no words sings on "aa" - akaaram - and the
                # voice renderer already does exactly that when handed no
                # lyrics.  The line is heard naked, before anyone decides
                # what should play it.
                ctx.progress(0.2, "Humming the tune")
                audio = self.providers.voice.render_vocal(
                    melody, None, self.current_voice(),
                    self.project.vocal_direction, sr,
                    total_seconds=total, seed=melody.seed,
                    # Every note, including the prelude and interlude: this
                    # is the tune being heard, not a take being sung over
                    # an arrangement.
                    vocal_sections_only=False,
                    # Closed, not open.  An open "aa" through four strong
                    # formants is close to how you would synthesise a reed
                    # instrument, which is what it sounded like.
                    vowel="hum")
                from .audio import dsp
                stereo = dsp.reverb(dsp.pan_mono(audio, 0.0), sr, 0.4, 0.16)
                stereo = dsp.normalize_loudness(stereo, sr, -17.0)
                return kind, dsp.limiter(stereo, -1.0, sr), {"hummed": True}

            def prog(p: float, msg: str) -> None:
                ctx.progress(0.1 + 0.8 * p, msg)

            result = mixer.mix(arrangement, vocal_audio, sr, total,
                               kind="full" if kind == "full" else kind,
                               vocal_gain=settings.vocal_gain,
                               settings=settings,
                               progress=prog, cancelled=lambda: ctx.cancelled)
            return kind, result.audio, {"summary": result.summary(),
                                        "notes": result.notes,
                                        "loudness": result.loudness_db,
                                        "tracks": result.track_count}

        render_ticket = self.song_work_ticket()

        def done(result: Tuple[str, np.ndarray, dict]) -> None:
            # Guarded before the file is written and before anything is
            # played: a mix of a song we have left would otherwise be saved
            # into the current song's folder and played back as its own.
            stale = self.stale_reason(render_ticket)
            if stale:
                self.status(f"I did not keep that mix: {stale}.")
                return
            rendered_kind, audio, info = result
            if rendered_kind == "tune":
                path = self._write_artifact("audio", f"tune_v{melody.version}.wav", audio)
                melody.audio_path = str(path)
            else:
                version = len(self.project.mixes) + 1
                path = self._write_artifact(
                    "mixes", f"{rendered_kind}_v{version}.wav", audio)
                mix_version = MixVersion(
                    version=version, kind=rendered_kind, audio_path=str(path),
                    duration=len(audio) / sr,
                    loudness_db=float(info.get("loudness", 0.0)),
                    arrangement_version=arrangement.version if arrangement else 0)
                self.project.mixes.append(mix_version)
                self.project.current_stage = Stage.MIX
            self._cache_render(rendered_kind, audio, str(path))
            for note in info.get("notes", []):
                log.info("mix note: %s", note)
            self._changed("render", f"Rendered the {rendered_kind}", undoable=False)
            self.status(f"{rendered_kind.replace('_', ' ').title()} ready"
                        + (f" - {info['summary']}" if info.get("summary") else ""))
            if autoplay:
                self.play_render(rendered_kind, play_range)

        self.jobs.submit(f"render.{kind}", f"render:{kind}", work, on_done=done,
                         on_error=lambda e: self.error("render", f"Render failed: {e}"),
                         description=f"Render the {kind}")

    def rendered(self, kind: str) -> Optional[RenderedAudio]:
        return self._renders.get(kind)

    def current_vocal_take(self):
        """The vocal take to play, and which render holds its audio.

        Newest wins.  The window asked for "vocal_master" first and
        "vocal_preview" only if there was none, which is a preference by
        kind rather than by recency: a master rendered an hour ago, to
        different words and in a different voice, outranked a preview
        rendered a moment before the creator pressed Play.
        """
        best_kind, best = None, None
        for kind in ("vocal_master", "vocal_preview"):
            rendered = self._renders.get(kind)
            if rendered is None:
                continue
            if best is None or rendered.created_at > best.created_at:
                best_kind, best = kind, rendered
        if best_kind is None:
            return None
        wanted = "master" if best_kind == "vocal_master" else "preview"
        take = next((t for t in reversed(self.project.vocal_renders)
                     if t.kind == wanted), None)
        return best_kind, take

    def play_vocal(self) -> bool:
        """Play the current vocal take, and say what is being played.

        Which take, in whose voice, and a word when that is not the voice
        now selected - a listening test on the wrong take is worse than no
        listening test, because it looks like an answer.
        """
        current = self.current_vocal_take()
        if current is None:
            self.status("Render a vocal take first.")
            return False
        kind, take = current
        singer = self.voices.get(take.voice_profile_id) if take else None
        who = singer.name if singer else "an unknown voice"
        what = "studio master" if kind == "vocal_master" else "preview"
        note = f"Playing the {what}"
        if take is not None:
            note += f" v{take.version}"
        note += f", sung by {who}."
        chosen = self.current_voice()
        if take is not None and singer is not None and singer.id != chosen.id:
            note += (f" This take was made before you chose {chosen.name}; "
                     f"render again to hear that voice.")
        # Said after playback starts, not before: play_render writes its own
        # line, so setting this first meant the creator saw "Playing vocal
        # preview" and never learned which take or whose voice.  A failure
        # keeps play_render's message, which is the one that matters then.
        started = self.play_render(kind)
        if started:
            self.status(note)
        return started

    def best_render(self) -> Optional[str]:
        for kind in ("full", "instrumental", "vocal_master", "vocal_preview", "tune"):
            if kind in self._renders:
                return kind
        return None

    def play_render(self, kind: Optional[str] = None,
                    play_range: Optional[Tuple[float, float]] = None,
                    loop: bool = False) -> bool:
        kind = kind or self.best_render()
        if kind is None:
            self.status("Nothing has been rendered yet.")
            return False
        rendered = self._renders.get(kind)
        if rendered is None:
            self.status(f"The {kind} has not been rendered yet.")
            return False
        # Compare the render, not its name.  The engine's ``source_name`` is
        # the *kind* - "audition", "tune", "full" - and a second audition is
        # still called "audition", so a name check said "already loaded" and
        # replayed the previous raaga's scale.  Auditioning Hamsadhwani then
        # Mohanam played Hamsadhwani twice; re-rendering a mix and pressing
        # play gave you the version before it.
        #
        # The guard is still worth having: ``load`` stops playback and
        # rewinds, so reloading the render already playing would break
        # pausing and playing a range of it.  Identity keeps that and fixes
        # the rest, because a re-render is always a new object.
        if self._loaded_render is not rendered \
                or self.playback.source_name != kind:
            self.playback.load(rendered.audio, rendered.sample_rate, kind)
            self._loaded_render = rendered
        start, end = (play_range or (None, None))
        ok = self.playback.play(start, end, loop)
        if not ok:
            # Nothing was heard, so the creator is still listening to
            # whatever they were listening to before.  Leaving the scope
            # alone is what keeps the comparison honest.
            self.error("playback", self.playback.last_error or "Playback failed.")
        else:
            # Every way of hearing the song arrives here - a fresh render,
            # a cached one, a section played from the table - so this is
            # the one place that knows what is actually being heard.
            span = (start, end) if start is not None and end is not None else None
            self._audition = ((span, self._span_name(self.project.melody(), span))
                              if span else None)
            self.status(f"Playing {kind.replace('_', ' ')}"
                        + (f" {start:.0f}-{end:.0f}s" if start is not None and end else ""))
        return ok

    def play_range(self, start: Optional[float], end: Optional[float],
                   loop: bool = False) -> bool:
        return self.play_render(None, (start, end) if start is not None else None, loop)

    def stop(self) -> None:
        """Stop is for stopping, and now that is all it is for.

        It used to be the way out of the last thing before starting the
        next one; the creative actions take the floor themselves now, so
        pressing this is a deliberate "silence".  Say which of the two
        happened rather than reporting "Stopped" at a quiet transport.
        """
        was_sounding = self.playback.playing or self.playback.paused
        self.playback.stop()
        self.status("Stopped" if was_sounding else "Nothing was playing")

    def pause(self) -> None:
        self.playback.pause()
        self.status("Paused")

    def resume(self) -> None:
        if not self.playback.resume():
            self.play_render()

    def seek(self, seconds: float) -> None:
        self._playhead = max(0.0, min(float(seconds),
                                      max(0.0, self.project.duration)))
        self.playback.seek(self._playhead)
        self.context.playhead = self.playhead

    def set_selection(self, start: Optional[float], end: Optional[float]) -> None:
        self.selection = (start, end) if start is not None and end is not None else None
        self._sync_context()

    # ==================================================================
    # export
    # ==================================================================
    def export(self, path: Path, kind: str = "full") -> Optional[Path]:
        rendered = self._renders.get(kind)
        if rendered is None:
            self.status(f"Render the {kind} before exporting it.")
            return None
        try:
            out = export_engine.export_audio(Path(path), rendered.audio,
                                             rendered.sample_rate)
        except Exception as exc:  # noqa: BLE001
            self.error("export", str(exc))
            return None
        self.project.log_history("export", f"Exported {kind} to {out}")
        self.status(f"Exported {kind} to {out}")
        return out

    def export_midi(self, path: Path) -> Optional[Path]:
        melody = self.project.melody()
        if melody is None:
            return None
        out = export_engine.write_midi(Path(path), melody, self.project.arrangement())
        self.status(f"Exported MIDI to {out}")
        return out

    def export_musicxml(self, path: Path) -> Optional[Path]:
        melody = self.project.melody()
        if melody is None:
            return None
        out = export_engine.write_musicxml(Path(path), melody,
                                           self.project.lyrics_version())
        self.status(f"Exported notation to {out}")
        return out

    def export_lyrics(self, path: Path) -> Optional[Path]:
        lyrics = self.project.lyrics_version()
        if lyrics is None:
            return None
        out = export_engine.write_lyrics_text(Path(path), lyrics, self.project.melody())
        self.status(f"Exported lyrics to {out}")
        return out

    def export_stems(self, directory: Path) -> List[Path]:
        arrangement = self.project.arrangement()
        if arrangement is None:
            return []
        stems = mixer.stems(arrangement, self.sample_rate, self.project.duration)
        out = export_engine.export_stems(Path(directory), stems, self.sample_rate)
        self.status(f"Exported {len(out)} stem(s)")
        return out

    def archive(self, path: Path) -> Optional[Path]:
        if self.project_dir is None:
            self.save()
        self.save()
        out = export_engine.archive_project(Path(path), self.project_dir)
        self.status(f"Project archived to {out}")
        return out

    def provider_statuses(self) -> List[ProviderStatus]:
        """Configured / Not configured / Unavailable / Ready ... (spec 41)."""
        return _provider_statuses(self.providers, self.settings,
                                  stt_adapter=self.voice_input.adapter)

    def _provider_status_table_text(self) -> str:
        rows = self.provider_statuses()
        header = f"{'Provider':<20}{'Kind':<10}{'State':<16}{'Model':<24}Detail"
        lines = [header, "-" * len(header)]
        for r in rows:
            lines.append(f"{r.name:<20}{r.kind:<10}{r.state:<16}{r.model:<24}{r.detail}")
        return "\n".join(lines)

    def export_diagnostics(self, path: Path) -> Path:
        return export_diagnostics(Path(path), self.project_dir,
                                  {"project": self.project.title,
                                   "stage": self.project.current_stage.value,
                                   "providers": self.providers.summary()},
                                  extra_files={"providers.txt":
                                              self._provider_status_table_text()})

    # ==================================================================
    # undo / redo
    # ==================================================================
    def undo_action(self) -> bool:
        result = self.undo.undo()
        if result is None:
            self.status("Nothing to undo.")
            return False
        project, label = result
        self.project = project
        self._renders.pop("full", None)
        self.status(f"Undid: {label}")
        self._sync_context()
        if self.on_project_changed:
            self.on_project_changed()
        self.dirty = True
        return True

    def redo_action(self) -> bool:
        result = self.undo.redo()
        if result is None:
            self.status("Nothing to redo.")
            return False
        project, label = result
        self.project = project
        self.status(f"Redid: {label}")
        self._sync_context()
        if self.on_project_changed:
            self.on_project_changed()
        self.dirty = True
        return True

    # ==================================================================
    # conversation
    # ==================================================================
    def start_listening(self) -> bool:
        ok = self.voice_input.start()
        self.context.listening = self.voice_input.state.listening
        self.status("Listening" if ok else self.voice_input.state.error)
        self._notify_conversation()
        return ok

    def stop_listening(self) -> None:
        """Stop, and mean it.

        The microphone stopping is not enough on its own: whatever was
        already heard is still queued, and before this the application
        would work through the whole backlog after being told to stop -
        which is how a conversation kept the interpreter busy long after
        the creator had pressed the button.
        """
        self.voice_input.stop()
        self.context.listening = False
        dropped = self._clear_utterances()
        self.jobs.cancel_target("voice")
        # Only what was heard.  A typed instruction interpreting right now
        # stays, and so does the flag that says something is in flight -
        # clearing it unconditionally would let a second phrase start
        # alongside the first.
        if not self.jobs.active_jobs():
            self._interpreting = None
            self._interpreting_id = 0
        if dropped:
            self.status(f"Microphone off - {dropped} unheard phrase(s) discarded")
        else:
            self.status("Microphone off")
        self._notify_conversation()

    def _clear_utterances(self) -> int:
        """Throw away anything heard but not yet acted on.

        Typed instructions are left alone: Stop Listening is about the
        microphone, and silently discarding what the creator typed would be
        a different act wearing the same button.
        """
        discarded = 0
        while True:
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                return discarded
            discarded += 1


    def toggle_listening(self) -> bool:
        return self.stop_listening() if self.context.listening else self.start_listening()

    def _on_transcript_partial(self, text: str) -> None:
        self.context.partial = text
        self._notify_conversation()

    def _on_barge_in(self) -> None:
        """The creator started speaking: interrupt speculative work at once."""
        if self.playback.playing:
            self.playback.pause()
        cancelled = self.jobs.cancel_all("interrupted by the creator")
        if cancelled:
            log.info("barge-in cancelled %d job(s)", cancelled)
            self.status("Listening - paused what I was doing")
        self._notify_conversation()

    def _on_transcript_final(self, text: str) -> None:
        """Called on the speech thread: queue it, do not act on it here.

        ``handle_utterance`` interprets the phrase, possibly through a
        language model, changes the project and asks the window to redraw.
        None of that may happen on a capture thread, so it waits for the
        next ``pump`` the way a finished job does.
        """
        self.context.partial = ""
        try:
            self._utterance_queue.put_nowait((text, self._project_generation))
        except queue.Full:
            # Never block here: this runs on the capture thread, and making
            # the microphone wait for the interpreter is how the backlog
            # became unbounded in the first place.
            self._utterances_dropped += 1
            log.warning("utterance queue full; dropped %r (%d dropped so far)",
                        text[:60], self._utterances_dropped)
            return
        self._notify_conversation()

    def _drain_utterances(self) -> None:
        """Start interpreting at most one phrase, and never on this thread.

        Two faults met here on 2026-09-06 and took the application down.

        This drained the *whole* queue on every pump, and each phrase was
        interpreted synchronously - which, when the rule tables do not
        recognise a phrase, means waiting for a language model.  ``pump``
        runs on the interface thread, so the window froze for seconds at a
        time while a queue of overheard conversation built up behind it.
        Stop Listening could not be serviced, because the thread that would
        service it was the thread that was busy.

        So: one phrase per pump, interpreted in the background, applied
        here when the answer arrives.  The interface stays alive whatever
        the room is saying.
        """
        if self._interpreting is not None:
            return                      # one in flight is enough
        # Typed first.  A deliberate instruction should not queue behind
        # whatever the room happened to say while it was being typed.
        origin = "typed"
        try:
            text, generation = self._typed_queue.get_nowait()
        except queue.Empty:
            origin = "voice"
            try:
                text, generation = self._utterance_queue.get_nowait()
            except queue.Empty:
                return
        if generation != self._project_generation:
            # Queued about a song that is no longer open.
            self.status(f"Not acting on {text[:40]!r}: it was about a "
                        f"different song.")
            return

        self._request_seq += 1
        request_id = self._request_seq
        self._interpreting = text
        self._interpreting_id = request_id
        self.status(f"Working on what you said: {text[:48]!r}")

        def release() -> None:
            """Give up the in-flight marker, but only if we still hold it."""
            if self._interpreting_id == request_id:
                self._interpreting = None
                self._interpreting_id = 0

        def work(ctx: JobContext) -> Command:
            self._sync_context()
            return interpret(text, self.context.time_context(),
                             llm=self.providers.llm,
                             last_instrument=self.context.last_instrument)

        def done(cmd: Command) -> None:
            release()
            if generation != self._project_generation:
                # The song changed while this was being interpreted.  It
                # would apply against whatever is open now, which is not
                # what was being talked about.
                self.status(f"Not acting on {text[:40]!r}: the song changed "
                            f"while I was working it out.")
                return
            try:
                self.apply_utterance(text, cmd)
            except Exception as exc:  # noqa: BLE001
                log.exception("could not act on what was heard: %s", text)
                self.error("voice", f"I could not act on that: {exc}")

        def failed(exc: Exception) -> None:
            release()
            log.warning("interpreting %r failed: %s", text[:60], exc)
            self.status(f"I could not work out what {text[:40]!r} meant.")

        # Typed instructions run under their own target.  Stop Listening
        # cancels the microphone's work, and a deliberate instruction that
        # happened to be interpreting at that moment is not the
        # microphone's work - it was cancelled with it, and the creator's
        # choice was simply never applied.
        self.jobs.submit(f"{origin}.interpret", origin, work,
                         on_done=done, on_error=failed,
                         on_cancelled=release,
                         description=("Interpret what was typed" if origin == "typed"
                                      else "Interpret what was heard"))

    def say(self, text: str) -> bool:
        """Take one typed instruction, and interpret it off this thread.

        The conversation box used to call ``handle_utterance`` directly,
        which interprets inline and may wait on a language model.  That is
        the same fault the microphone had on 2026-09-06 - the window froze
        because the thread that would redraw it was the thread doing the
        interpreting - and I left it in place for typed text when I fixed
        the microphone this morning.  One door now, for both.

        Returns False only when the typed backlog is genuinely full, and
        says so rather than dropping the instruction silently.
        """
        text = (text or "").strip()
        if not text:
            return False
        try:
            self._typed_queue.put_nowait((text, self._project_generation))
        except queue.Full:
            self.status("I am still working through what you have already "
                        "said. Give me a moment.")
            return False
        self._notify_conversation()
        return True

    def handle_utterance(self, text: str) -> Command:
        """Interpret one instruction and act on it, inline.

        Kept for the conversational path and for tests, where a synchronous
        answer is simpler.  The microphone does *not* use this: it goes
        through ``_drain_utterances``, which interprets in the background so
        that a room full of talking cannot freeze the window.
        """
        self._sync_context()
        cmd = interpret(text, self.context.time_context(), llm=self.providers.llm,
                        last_instrument=self.context.last_instrument)
        return self.apply_utterance(text, cmd)

    def apply_utterance(self, text: str, cmd: Command) -> Command:
        """Act on an already-interpreted phrase.  Fast, and UI-thread safe."""
        cmd = self.context.resolve(cmd)
        turn = self.context.add_turn(text, intent=cmd.intent,
                                     interpretation=cmd.interpretation)
        self.project.conversation.append(turn)
        self._notify_conversation()

        # An instrument the catalog does not have is reported by name with the
        # closest alternatives, never silently substituted (spec 7.1).
        asked_for_an_instrument = cmd.intent in ("arrange.add", "arrange.replace")
        if not cmd.known or (asked_for_an_instrument and not cmd.instrument
                             and not cmd.feel_words):
            missing = unavailable_instrument(text)
            if missing:
                phrase, alternatives = missing
                reason = (f"I do not have a '{phrase}'. Closest available: "
                          f"{', '.join(alternatives)}.")
                self.context.update_status(turn.id, "failed",
                                           action="add an instrument",
                                           reason=reason)
                self.error("arrangement", reason)
                return cmd
        if not cmd.known:
            # "Not understood" is a result, and saying which part defeated
            # it is the difference between a report and a shrug.
            reason = (f"I could not tell what to do with {text!r}. Try naming "
                      f"the action - generate a tune, add a violin, slower.")
            self.context.update_status(turn.id, "ignored", reason=reason)
            self.status(f"I did not understand: {text!r}")
            return cmd

        action = describe(cmd) or cmd.intent
        try:
            self.execute(cmd)
            self.context.update_status(turn.id, "applied", action=action,
                                       reason=self.status_text)
        except LockedContentError as exc:
            self.context.update_status(turn.id, "failed", action=action,
                                       reason=str(exc))
            self.error("locked", str(exc))
        except Exception as exc:  # noqa: BLE001
            self.context.update_status(turn.id, "failed", action=action,
                                       reason=f"{cmd.intent} failed: {exc}")
            self.error("command", f"{cmd.intent} failed: {exc}")
        self.context.remember(cmd)
        self._notify_conversation()
        return cmd

    def execute(self, cmd: Command) -> None:
        """Apply an interpreted command. The newest instruction always wins."""
        spec: Optional[TimeSpec] = cmd.time
        start = spec.start if spec else None
        end = spec.end if spec else None
        intent = cmd.intent

        if intent == "transport.play":
            self.play_range(start, end)
        elif intent == "transport.pause":
            self.pause()
        elif intent == "transport.stop":
            self.stop()
        elif intent == "transport.resume":
            self.resume()
        elif intent == "transport.loop":
            self.play_range(start, end, loop=True)
        elif intent == "transport.seek":
            if spec and spec.relative is not None:
                self.seek(self.playhead + spec.relative)
            elif start is not None:
                self.seek(start)
            self.status(f"Playhead at {self.playhead:.1f}s")

        elif intent == "arrange.add":
            if not cmd.instrument:
                self.status("Which instrument would you like?")
                return
            self.add_instrument(cmd.instrument, start or 0.0,
                                end or self.project.duration)
        elif intent == "arrange.remove":
            self.remove_instrument(cmd.instrument, start, end)
        elif intent == "arrange.replace":
            if not cmd.target_instrument:
                self.status("Replace it with which instrument?")
                return
            self.replace_instrument(cmd.instrument, cmd.target_instrument, start, end)
        elif intent == "arrange.suggest":
            ranked = self.suggest_instruments(cmd.feel_words or self.context.last_feel_words)
            if not ranked:
                self.status("No instrument in the catalog matches that feel.")
                return
            best = ranked[0][0]
            others = ", ".join(i.name for i, _ in ranked[1:3])
            self.add_instrument(best.key, start or 0.0, end or self.project.duration)
            self.status(f"Trying {best.name} for that feel"
                        + (f" (alternatives: {others})" if others else ""))
        elif intent == "arrange.level":
            self.change_level(cmd.instrument, cmd.value or 0.85, start, end)
        elif intent in ("arrange.mute", "arrange.solo"):
            arrangement = self.project.arrangement()
            if arrangement is None or not cmd.instrument:
                return
            for track in arrangement.tracks_for_instrument(cmd.instrument):
                self.set_track_flag(track.id,
                                    mute=True if intent == "arrange.mute" else None,
                                    solo=True if intent == "arrange.solo" else None)
        elif intent == "arrange.regenerate":
            if self.context.last_track_id and self.context.last_region_id:
                self.regenerate_region(self.context.last_track_id,
                                       self.context.last_region_id)
            else:
                self.auto_arrange()
        elif intent == "arrange.auto":
            self.auto_arrange()

        elif intent == "tune.generate":
            # "Compose a tune in Hamsadhwani" parsed the raaga correctly and
            # then composed in whatever was already selected, because this
            # branch never read cmd.raaga.  The raaga a creator names in the
            # request is the raaga they are asking to hear.
            if self._compose_in_named_raaga(cmd):
                self.generate_tune()
        elif intent == "tune.variation":
            self.make_variation()
        elif intent == "tune.accept":
            self.accept_tune()
        elif intent == "tune.regenerate_section":
            section_id = cmd.section_id or self.context.last_section_id
            if section_id:
                self.regenerate_tune_section(section_id)
            else:
                self.status("Which section should I rewrite?")
        elif intent == "tune.tempo":
            melody = self.project.melody()
            if cmd.value and cmd.value > 20:
                self.set_tempo(int(cmd.value))
            elif cmd.value and melody:
                self.set_tempo(int(melody.tempo_bpm * cmd.value))

        elif intent == "lyrics.generate":
            self.generate_lyrics()
        elif intent == "lyrics.accept":
            self.accept_lyrics()

        elif intent == "raaga.set":
            if cmd.raaga:
                self.select_raaga(cmd.raaga)
                self.status(f"Raaga set to {cmd.raaga}")
        elif intent == "raaga.suggest":
            names = ", ".join(s.name for s in self.raaga_suggestions())
            self.status(f"Suggested raagas: {names}")
        elif intent == "raaga.lock":
            self.set_raaga_lock(True)

        elif intent == "voice.render":
            self.render_vocal("preview")
        elif intent == "voice.vocal_only":
            self.render_vocal("master")
        elif intent == "voice.direction":
            if cmd.style:
                self.set_vocal_direction(style=cmd.style)
        elif intent == "voice.set":
            profile = self.voices.by_name(cmd.text)
            if profile:
                self.set_voice(profile.id)
                self.status(f"Singer: {profile.name}")

        elif intent == "mix.full":
            self.render("full", autoplay=False)
        elif intent == "mix.instrumental":
            self.render("instrumental", autoplay=False)
        elif intent == "mix.export":
            self.status("Use File > Export to choose a destination.")

        elif intent == "region.lock":
            if start is not None:
                self.lock_range(start, end or self.project.duration, True)
        elif intent == "region.unlock":
            if start is not None:
                self.lock_range(start, end or self.project.duration, False)

        elif intent == "agent.learn":
            named = self.raagas.find_in_text(cmd.text)
            if named:
                self.study_raaga(named.name)
                self.learn_now(1)
            else:
                self.learn_now(2)
        elif intent == "agent.explain":
            self.status(self.ask_agent(cmd.text).splitlines()[0][:200])
        elif intent == "agent.feedback":
            self.give_feedback(cmd.text)
        elif intent == "agent.status":
            state = self.agent_status()
            self.status(f"Stage {state['stage']}, studying "
                        f"{state['current_raaga']}: {state['next_goal']}")

        elif intent == "project.save":
            self.save()
        elif intent == "project.undo":
            self.undo_action()
        elif intent == "project.redo":
            self.redo_action()
        elif intent == "project.cancel":
            self.jobs.cancel_all("cancelled by the creator")
            self.status("Cancelled the current operation")

    # ==================================================================
    # reporting
    # ==================================================================
    def summary(self) -> str:
        p = self.project
        melody = p.melody()
        rows = [f"Project: {p.title}",
                f"Stage:   {p.current_stage.value}",
                f"Brief:   {p.brief.summary()}",
                f"Raaga:    {p.raaga.selected or '-'}"
                f"{' (locked)' if p.raaga.locked else ''}"]
        if melody:
            rows.append(f"Tune:    v{melody.version}, {melody.tempo_bpm} bpm, "
                        f"{melody.duration:.0f}s, {len(melody.notes)} notes "
                        f"[{melody.state.value}]")
        lyrics = p.lyrics_version()
        if lyrics:
            rows.append(f"Lyrics:  v{lyrics.version}, {len(lyrics.lines)} lines")
        take = p.vocal_master or p.latest_vocal
        if take:
            rows.append(f"Vocal:   {take.kind} take v{take.version}")
        arrangement = p.arrangement()
        if arrangement:
            rows.append(f"Tracks:  " + ", ".join(t.label for t in arrangement.tracks))
        if p.mixes:
            rows.append(f"Mixes:   {len(p.mixes)}")
        return "\n".join(rows)
