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
import queue
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
from .core.actions import ActionState, ActionStatus
from .core.jobs import JobCancelled, JobContext, JobManager
from .core.logging_setup import export_diagnostics, get_logger, setup_logging
from .core.models import (ApprovalState, ArrangementVersion, CreativeBrief,
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
from .music.structure import plan_sections
from .music.synth import render_notes
from .music.validator import validate
from .providers import registry as provider_registry
from .providers.status import ProviderStatus
from .providers.status import provider_statuses as _provider_statuses
from .raaga import audition
from .raaga.library import Raaga, library as raaga_library
from .raaga.selection import (RaagaSuggestion, expand_feel_words, infer_tempo,
                              suggest as suggest_raagas)
from .speech.capture import CaptureState, VoiceInputManager
from .speech.context import ConversationContext
from .speech.intent import Command, interpret, unavailable_instrument
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
        self.last_evaluation = None

        # The action status contract (v0.3 section 6.1).  ``actions`` holds
        # the latest status per action name; ``_action_queue`` carries
        # statuses raised on a background job's worker thread across to the
        # UI thread, the same way JobManager carries job results (see
        # ``core/jobs.py`` and ``pump`` below).
        self.actions: Dict[str, ActionStatus] = {}
        self._action_queue: "queue.Queue[ActionStatus]" = queue.Queue()
        self.last_suggestions: List = []
        #: The brief ``last_suggestions`` were made for, so selection
        #: feedback is attached to what was actually asked.
        self.suggested_for: Optional[CreativeBrief] = None

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
        self._sync_context()
        if events and self.on_project_changed:
            self.on_project_changed()
        self.maybe_autosave()

    # ==================================================================
    # project lifecycle
    # ==================================================================
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

        job = self.jobs.submit(
            "brief.apply", "brief", work,
            on_done=self._apply_brief_done,
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
        self.project.raaga.alternatives = [s.name for s in suggestions]
        # The brief these answer, kept as it was at the time.  Selection
        # feedback has to be attached to the brief the suggestions were made
        # for, not to whatever is in the panel when the creator gets round to
        # choosing: apply a brief, apply a second one, then pick from the
        # first list, and the agent would otherwise learn that the raaga
        # suits a feeling nobody was asking about.
        self.suggested_for = replace(self.project.brief)

    def _apply_brief_done(self, result: Tuple[ActionStatus, List]) -> None:
        status, suggestions = result
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

    def tune_instrument(self):
        """Which instrument the tune and the audition are heard on.

        The creative brief already has a "Prefer" field, and until now
        nothing read it: the tune was rendered on a hardcoded veena whatever
        the creator asked for.  So the brief decides, then the setting, then
        the veena that used to be the only answer.

        Only an instrument that can carry a lead is taken from the brief - a
        mridangam listed under "prefer" is a real preference about the
        arrangement and not an offer to play the melody on it.
        """
        for name in self.project.brief.instruments_preferred:
            instrument = catalog.get(str(name).strip())
            if instrument is not None and "lead" in instrument.roles:
                return instrument
        configured = catalog.get(getattr(self.settings, "tune_instrument", ""))
        if configured is not None and "lead" in configured.roles:
            return configured
        return catalog.get("veena") or catalog.all_instruments()[0]

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
        tempo = infer_tempo(brief, raaga)
        melody = self.project.melody()
        return MelodyOptions(
            tempo_bpm=melody.tempo_bpm if melody else tempo,
            beats_per_cycle=melody.beats_per_cycle if melody else 8,
            tonic_midi=self.project.raaga.tonic_midi,
            voice_low=profile.range_low, voice_high=profile.range_high,
            intensity=0.6, seed=seed if seed is not None else int(time.time()) % 9999,
            song_type=brief.song_type, duration_target=brief.duration_target)

    def generate_tune(self, seed: Optional[int] = None) -> None:
        raaga = self.composing_raaga()
        opts = self.melody_options(seed)
        opts.tempo_bpm = infer_tempo(self.project.brief, raaga)
        version = (max((m.version for m in self.project.melodies), default=0)) + 1
        self.status(f"Composing a tune in {raaga.name}...")

        brief = self.project.brief
        project_id = self.project.project_id
        max_rewrites = max(0, int(getattr(self.settings, "compose_rewrites", 3)))
        threshold = float(getattr(self.settings, "compose_threshold", 0.7))
        bank = self.agent.phrase_bank(raaga.name)

        def work(ctx: JobContext) -> MelodyVersion:
            ctx.progress(0.15, "Planning sections")
            sections = plan_sections(opts.duration_target, opts.tempo_bpm,
                                     opts.beats_per_cycle, opts.song_type)

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
            best.guidance_note = guidance_note
            return best

        self.jobs.submit("tune.generate", "melody:all", work,
                         on_done=lambda m: self._tune_ready(m, "Generated"),
                         on_error=lambda e: self.error("tune", f"Tune generation failed: {e}"),
                         description=f"Compose a tune in {raaga.name}")

    def make_variation(self, strength: float = 0.5) -> None:
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

        self.jobs.submit("tune.variation", "melody:all", work,
                         on_done=lambda m: self._tune_ready(m, "Variation"),
                         on_error=lambda e: self.error("tune", f"Variation failed: {e}"),
                         description="Tune variation")

    def regenerate_tune_section(self, section_id: str) -> None:
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

        self.jobs.submit("tune.section", f"melody:{section_id}", work,
                         on_done=lambda m: self._tune_ready(m, f"Rewrote {section.name}"),
                         on_error=lambda e: self.error("tune", f"Section rewrite failed: {e}"),
                         description=f"Rewrite {section.name}")

    def set_tempo(self, bpm: int) -> None:
        melody = self.project.melody()
        if melody is None:
            self.project.brief.tempo_preference = int(bpm)
            self._changed("tune.tempo", f"Tempo preference {bpm} bpm")
            return
        version = max(m.version for m in self.project.melodies) + 1
        fresh = melody_engine.retempo(melody, int(bpm), version)
        self._tune_ready(fresh, f"Tempo {bpm} bpm")

    def _tune_ready(self, melody: MelodyVersion, what: str) -> None:
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

        self.status(f"{what} tune v{melody.version} "
                    f"({melody.duration:.0f}s, {len(melody.notes)} notes)"
                    f"{critique}")
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

    def ask_agent(self, question: str) -> str:
        raaga = self.project.raaga.selected or ""
        melody = self.project.melody()
        low = (question or "").lower()
        if melody is not None and ("why" in low or "phrase" in low):
            return self.agent.explain_choice(melody, raaga or melody.raaga)
        return self.agent.explain(question, raaga)

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
        return validate(melody, raaga).summary()

    # ==================================================================
    # lyrics
    # ==================================================================
    def generate_lyrics(self, seed: Optional[int] = None) -> None:
        melody = self.project.melody()
        if melody is None:
            self.status("Write a tune first - the lyrics are fitted to it.")
            return
        brief = self.project.brief
        version = max((l.version for l in self.project.lyrics), default=0) + 1
        previous = self.project.lyrics_version()
        llm = self.providers.llm
        self.status("Writing lyrics to fit the tune...")

        def work(ctx: JobContext) -> LyricsVersion:
            ctx.progress(0.3, "Measuring phrases")
            return lyric_generator.generate(
                melody, brief, version=version,
                seed=seed if seed is not None else version * 17, llm=llm,
                previous=previous)

        def done(lyrics: LyricsVersion) -> None:
            self.project.lyrics.append(lyrics)
            self.project.approved_lyrics = lyrics.version
            self.project.current_stage = Stage.VOICE
            self._changed("lyrics.version", f"Lyrics v{lyrics.version}")
            self.status(f"Lyrics v{lyrics.version}: {len(lyrics.lines)} lines fitted")

        self.jobs.submit("lyrics.generate", "lyrics", work, on_done=done,
                         on_error=lambda e: self.error("lyrics", f"Lyrics failed: {e}"),
                         description="Write lyrics for the tune")

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

    def render_vocal(self, kind: str = "preview", autoplay: bool = True) -> None:
        """kind is 'preview' or 'master' (the studio vocal-only version)."""
        melody = self.project.melody()
        if melody is None:
            self.status("There is no tune to sing yet.")
            return
        lyrics = self.project.lyrics_version()
        profile = self.current_voice()
        direction = self.project.vocal_direction
        sr = self.sample_rate
        provider = self.providers.voice
        version = len(self.project.vocal_renders) + 1
        self.status("Rendering the vocal..." if kind == "preview"
                    else "Producing the studio vocal-only master...")

        def work(ctx: JobContext) -> Tuple[VocalRender, np.ndarray]:
            ctx.progress(0.2, "Singing the line")
            raw = provider.render_vocal(melody, lyrics, profile, direction, sr,
                                        melody.duration + 1.0, seed=version * 7)
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
            self.status(f"Vocal {kind} ready - {mastering.report(audio, self.sample_rate)}")
            if autoplay:
                self.play_render("vocal_master" if kind == "master" else "vocal_preview")

        self.jobs.submit(f"voice.{kind}", "vocal", work, on_done=done,
                         on_error=lambda e: self.error("voice", f"Vocal render failed: {e}"),
                         description=f"Render the {kind} vocal")

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
        melody = self.project.melody()
        if melody is None:
            self.status("Write a tune before arranging.")
            return
        raaga = self.require_raaga()
        brief = self.project.brief
        previous = self.project.arrangement()
        self.status("Building a first arrangement...")

        def work(ctx: JobContext) -> ArrangementVersion:
            ctx.progress(0.3, "Choosing instruments")
            return arranger.auto_arrange(melody, raaga, brief, previous=previous)

        def done(arrangement: ArrangementVersion) -> None:
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
        if melody is None:
            self.status("Nothing to render yet.")
            return
        sr = self.sample_rate
        total = max(melody.duration, self.project.duration) + 0.5
        arrangement = self.project.arrangement()
        vocal_take = (self.project.vocal_master or self.project.latest_vocal)
        vocal_audio = None
        if kind in ("full", "vocal_only"):
            cached = self._renders.get("vocal_master") or self._renders.get("vocal_preview")
            if cached is not None:
                vocal_audio = cached.audio
        raaga = self.require_raaga()
        provider = self.providers.music
        self.status(f"Rendering the {kind.replace('_', ' ')}...")

        def work(ctx: JobContext) -> Tuple[str, np.ndarray, dict]:
            if kind == "tune":
                ctx.progress(0.2, "Sounding the tune")
                inst = self.tune_instrument()
                audio = provider.render_part(melody.notes, inst.key, sr,
                                             total_seconds=total, seed=melody.seed)
                from .audio import dsp
                stereo = dsp.reverb(dsp.pan_mono(audio, 0.0), sr, 0.4, 0.16)
                stereo = dsp.normalize_loudness(stereo, sr, -17.0)
                return kind, dsp.limiter(stereo, -1.0, sr), {}

            def prog(p: float, msg: str) -> None:
                ctx.progress(0.1 + 0.8 * p, msg)

            result = mixer.mix(arrangement, vocal_audio, sr, total,
                               kind="full" if kind == "full" else kind,
                               progress=prog, cancelled=lambda: ctx.cancelled)
            return kind, result.audio, {"summary": result.summary(),
                                        "notes": result.notes,
                                        "loudness": result.loudness_db,
                                        "tracks": result.track_count}

        def done(result: Tuple[str, np.ndarray, dict]) -> None:
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
            self.error("playback", self.playback.last_error or "Playback failed.")
        else:
            self.status(f"Playing {kind.replace('_', ' ')}"
                        + (f" {start:.0f}-{end:.0f}s" if start is not None and end else ""))
        return ok

    def play_range(self, start: Optional[float], end: Optional[float],
                   loop: bool = False) -> bool:
        return self.play_render(None, (start, end) if start is not None else None, loop)

    def stop(self) -> None:
        self.playback.stop()
        self.status("Stopped")

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
        self.voice_input.stop()
        self.context.listening = False
        self.status("Microphone off")
        self._notify_conversation()

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
        self.context.partial = ""
        self.handle_utterance(text)

    def handle_utterance(self, text: str) -> Command:
        """Interpret one instruction and act on it."""
        self._sync_context()
        cmd = interpret(text, self.context.time_context(), llm=self.providers.llm,
                        last_instrument=self.context.last_instrument)
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
                self.context.update_status(turn.id, "failed")
                self.error("arrangement",
                           f"I do not have a '{phrase}'. Closest available: "
                           f"{', '.join(alternatives)}.")
                return cmd
        if not cmd.known:
            self.context.update_status(turn.id, "ignored")
            self.status(f"I did not understand: {text!r}")
            return cmd

        try:
            self.execute(cmd)
            self.context.update_status(turn.id, "applied")
        except LockedContentError as exc:
            self.context.update_status(turn.id, "failed")
            self.error("locked", str(exc))
        except Exception as exc:  # noqa: BLE001
            self.context.update_status(turn.id, "failed")
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
