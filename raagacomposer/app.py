"""Application controller (spec section 12.2).

Everything the UI does goes through here: project lifecycle, the creative
workflow, background jobs, playback, voice commands, undo and export.  The UI
holds no musical logic; it renders this object's state and calls its methods.

Threading rule: worker functions compute and return, they never mutate the
project.  Completion callbacks run on the UI thread via
:meth:`JobManager.drain`, and only they write to project state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import export as export_engine
from .audio.playback import PlaybackEngine
from .core.jobs import JobContext, JobManager
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
from .music.validator import validate
from .providers import registry as provider_registry
from .raaga.library import Raaga, library as raaga_library
from .raaga.selection import RaagaSuggestion, infer_tempo, suggest as suggest_raagas
from .speech.capture import CaptureState, VoiceInputManager
from .speech.context import ConversationContext
from .speech.intent import Command, interpret, unavailable_instrument
from .speech.timeline_parser import TimeSpec
from .voice import mastering
from .voice.profiles import VoiceProfileManager

log = get_logger("app")

RENDER_KINDS = ("tune", "vocal_preview", "vocal_master", "instrumental", "full")


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

        self.project: Project = Project()
        self.project_dir: Optional[Path] = None
        self.dirty = False
        self._last_autosave = time.time()
        self._renders: Dict[str, RenderedAudio] = {}
        self.status_text = "Ready"
        self.selection: Optional[Tuple[float, float]] = None
        self._playhead = 0.0

        # UI callbacks
        self.on_project_changed: Optional[Callable[[], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_conversation: Optional[Callable[[], None]] = None
        self.on_render: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

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
        target = self.store.save_as(self.project, self.project_dir, Path(directory))
        self.project_dir = target
        self.dirty = False
        self.status(f"Saved a copy to {target}")
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

    # ==================================================================
    # creative brief and raaga
    # ==================================================================
    def update_brief(self, **fields) -> None:
        brief = self.project.brief
        for key, value in fields.items():
            if hasattr(brief, key):
                setattr(brief, key, value)
        self.project.current_stage = Stage.RAAGA if brief.summary() else Stage.BRIEF
        self._changed("brief.update", "Updated the creative brief")

    def raaga_suggestions(self, limit: int = 4) -> List[RaagaSuggestion]:
        suggestions = suggest_raagas(self.project.brief, self.raagas, limit=limit)
        llm = self.providers.llm
        if llm is not None and llm.available:
            try:
                names = [s.name for s in suggestions] or self.raagas.names()[:12]
                extra = llm.suggest_raagas(self.project.brief, names)
                order = {str(e.get("raaga", "")).lower(): str(e.get("reason", ""))
                         for e in extra}
                for s in suggestions:
                    if s.name.lower() in order:
                        s.rationale = order[s.name.lower()] or s.rationale
                suggestions.sort(key=lambda s: (s.name.lower() not in order, -s.score))
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM raaga advice failed: %s", exc)
        self.project.raaga.alternatives = [s.name for s in suggestions]
        return suggestions

    def select_raaga(self, name: str, rationale: str = "") -> Raaga:
        raaga = self.raagas.get(name)
        if raaga is None:
            raise KeyError(f"Unknown raaga: {name}")
        if self.project.raaga.locked:
            raise LockedContentError(
                f"The raaga is locked to {self.project.raaga.selected}. Unlock to change it.")
        self.project.raaga.selected = raaga.name
        self.project.raaga.rationale = rationale
        self.project.raaga.state = ApprovalState.APPROVED
        self.project.raaga.version += 1
        self.project.current_stage = Stage.TUNE
        self._changed("raaga.select", f"Selected raaga {raaga.name}")
        return raaga

    def set_raaga_lock(self, locked: bool) -> None:
        self.project.raaga.state = ApprovalState.LOCKED if locked else ApprovalState.APPROVED
        self._changed("raaga.lock", f"{'Locked' if locked else 'Unlocked'} the raaga")

    def current_raaga(self) -> Optional[Raaga]:
        return self.raagas.get(self.project.raaga.selected)

    def require_raaga(self) -> Raaga:
        raaga = self.current_raaga()
        if raaga is None:
            suggestions = self.raaga_suggestions(1)
            raaga = self.select_raaga(suggestions[0].name, suggestions[0].rationale)
        return raaga

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
        raaga = self.require_raaga()
        opts = self.melody_options(seed)
        opts.tempo_bpm = infer_tempo(self.project.brief, raaga)
        version = (max((m.version for m in self.project.melodies), default=0)) + 1
        self.status(f"Composing a tune in {raaga.name}...")

        def work(ctx: JobContext) -> MelodyVersion:
            ctx.progress(0.15, "Planning sections")
            sections = plan_sections(opts.duration_target, opts.tempo_bpm,
                                     opts.beats_per_cycle, opts.song_type)
            ctx.progress(0.45, "Writing phrases")
            melody = melody_engine.generate(raaga, opts, sections, version=version)
            ctx.progress(0.85, "Checking raaga fidelity")
            report = validate(melody, raaga, opts.voice_low, opts.voice_high)
            melody.validation = report.issues
            return melody

        self.jobs.submit("tune.generate", "melody:all", work,
                         on_done=lambda m: self._tune_ready(m, "Generated"),
                         on_error=lambda e: self.error("tune", f"Tune generation failed: {e}"),
                         description=f"Compose a tune in {raaga.name}")

    def make_variation(self, strength: float = 0.5) -> None:
        melody = self.project.melody()
        if melody is None:
            return self.generate_tune()
        raaga = self.require_raaga()
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
        raaga = self.require_raaga()
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
        self.status(f"{what} tune v{melody.version} "
                    f"({melody.duration:.0f}s, {len(melody.notes)} notes)")
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
                inst = catalog.get("veena") or catalog.all_instruments()[0]
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
        if self.playback.source_name != kind:
            self.playback.load(rendered.audio, rendered.sample_rate, kind)
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

    def export_diagnostics(self, path: Path) -> Path:
        return export_diagnostics(Path(path), self.project_dir,
                                  {"project": self.project.title,
                                   "stage": self.project.current_stage.value,
                                   "providers": self.providers.summary()})

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
