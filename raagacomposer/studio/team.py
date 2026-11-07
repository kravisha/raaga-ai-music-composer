"""The song team's agents, and the producer that runs them.

Four agents, each a small class with one job and one card to write:

* ``MoodListener`` reads the producer's words and writes a ``MoodCard``.
* ``Lyricist`` writes two lines from the mood card alone.
* ``Tunesmith`` picks the raaga and writes a short alapana and a two-line
  phrase, from the mood card alone - at the same time as the lyricist.
* ``TalaKeeper`` reads the tune and the words and picks the tala, then lays
  a beat under the phrase.

``SongTeam`` is the producer's desk: it runs them in that order, lyricist
and tunesmith side by side, fits the words onto the phrase (asking the
lyricist to rewrite a line that will not fit), and records every hand-over.

None of this touches the open project.  Each agent is built on an engine the
composer already trusts - ``raaga.selection``, ``music.melody``,
``lyrics.fitting``, ``music.tala``, ``music.beat`` - so the song team and the
composer cannot disagree about what a raaga or a tala is.
"""
from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..core.logging_setup import get_logger
from ..core.models import (
    BeatVersion,
    CreativeBrief,
    MelodyVersion,
    Note,
    Section,
    SectionKind,
)
from ..lyrics import generator as lyric_engine
from ..lyrics.fitting import PhraseSlot, build_slots, count_syllables, fit_lines
from ..music import beat as beat_engine
from ..music import tala as tala_module
from ..music.melody import MelodyOptions
from ..music.melody import generate as compose_melody
from ..music.theory import cycle_seconds
from ..raaga import selection
from ..raaga.library import Raaga, RaagaLibrary, library
from .handoff import (
    BeatProposal,
    LyricSheet,
    MoodCard,
    SongDraft,
    TeamEvent,
    TuneProposal,
)

log = get_logger("studio")

MOOD_LISTENER = "Mood listener"
LYRICIST = "Lyricist"
TUNESMITH = "Tunesmith"
TALA_KEEPER = "Tala keeper"
PRODUCER = "Producer"

#: The two lines of the simple path.
LINES = 2
#: Roughly how many beats one line should last, whatever the cycle: two
#: cycles of Adi, four of Tisra Eka.  A fixed count of cycles made a
#: three-beat lullaby line last two seconds.
BEATS_PER_LINE = 12


def cycles_per_line(beats_per_cycle: int) -> int:
    return max(1, round(BEATS_PER_LINE / max(1, beats_per_cycle)))

#: Said on every draft, because none of it is built yet.  Krish's
#: "condiments", plus what the rough picture itself still lacks.
NOT_DONE_YET: tuple[str, ...] = (
    "Prelude, fillers and bridge: not made yet (the rich path comes later).",
    "A strong start and a strong end: not made yet.",
    ("More instruments: only voice, tanpura and mridangam play in the rough "
    "picture.  Agents that each 'pick an instrument' and join: not built."),
    ("The singer is the built-in formant synthesiser - it shows the tune and "
    "where the words fall, it is not a real voice."),
    ("Without Claude or a large local model the lyricist uses the built-in "
    "word lists: lines fit the tune but are not poetry."),
    ("The tala keeper reads the mood words and the tempo; it does not yet "
    "listen to the phrase's shape."),
    ("Sending a draft into the composer (MAIN tabs) for full editing: not "
    "built yet."),
)

LANGUAGES = ("Tamil", "Telugu", "Hindi", "English")

#: Pace words the listener reports, by tempo.
_PACES = ((66, "slow"), (90, "medium"), (999, "lively"))


def _pace(bpm: int) -> str:
    return next(name for top, name in _PACES if bpm <= top)


# ==========================================================================
# mood
# ==========================================================================
class MoodListener:
    """Hears the producer's words and says back what it understood."""

    name = MOOD_LISTENER

    def listen(self, text: str, language: str = "") -> MoodCard:
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("Describe the mood in a few words first.")
        language = self._language(text, language)
        # The brief is built from the words alone.  A new project's default
        # situation and mood are a novice's nerves; carrying those in would
        # make every song the team writes nervous.
        brief = CreativeBrief(title="", situation="", mood=text, feel="",
                              notes="", language=language,
                              duration_target=30.0)
        feel_words = selection.expand_feel_words(text)
        themes = [t for t in lyric_engine._themes_for(brief)
                  if t != "connective"]
        tempo = selection.infer_tempo(brief, None)
        pace = _pace(tempo)
        # Slower songs carry fewer syllables a line.  This is a starting
        # point both writers agree on; the fit afterwards is the real test.
        per_line = {"slow": 7, "medium": 9, "lively": 11}[pace]
        return MoodCard(text=text, language=language, feel_words=feel_words,
                        themes=themes, tempo_bpm=int(tempo), pace=pace,
                        line_plan=[per_line] * LINES, brief=brief)

    @staticmethod
    def _language(text: str, fallback: str) -> str:
        low = text.lower()
        for lang in LANGUAGES:
            if re.search(rf"\b{lang.lower()}\b", low):
                return lang
        return fallback or "Tamil"


# ==========================================================================
# lyrics
# ==========================================================================
def _plan_slots(targets: Sequence[int]) -> list[PhraseSlot]:
    """Phrase slots with only a syllable count: what a lyricist is asked
    for before any tune exists.  The language-model prompt reads
    ``section_name``, ``syllable_count`` and ``stresses``, nothing else."""
    return [PhraseSlot(section_id="pallavi", section_name="Pallavi",
                       note_indices=list(range(n)), durations=[0.5] * n,
                       stresses=[i == 0 for i in range(n)])
            for n in targets]


class Lyricist:
    """Writes words from the mood.  Never sees the tune while writing."""

    name = LYRICIST

    def __init__(self, llm=None) -> None:
        self.llm = llm

    def write(self, mood: MoodCard, seed: int = 3) -> LyricSheet:
        lines: list[str] = []
        by = "built-in word lists"
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                lines = [str(x) for x in
                         self.llm.write_lyrics(_plan_slots(mood.line_plan),
                                               mood.brief)][:LINES]
                by = getattr(self.llm, "name", "a language model")
            except Exception as exc:  # noqa: BLE001 - fall back, never block
                log.warning("lyricist model failed (%s); using the word lists",
                            exc)
                lines = []
        # A line the singer cannot count is not a line (the same rule as
        # lyrics/generator.py): replace it from the word lists.
        local = lyric_engine.generate_lines(_plan_slots(mood.line_plan),
                                            mood.brief, seed)
        final: list[str] = []
        for i in range(LINES):
            if i < len(lines) and count_syllables(lines[i]) > 0:
                final.append(lines[i])
            else:
                final.append(local[i])
        if lines and final != lines[:LINES]:
            by += " (some lines from the word lists)"
        return LyricSheet(lines=final, language=mood.language, written_by=by)

    def revise(self, line: str, targets: Sequence[int], mood: MoodCard,
               seed: int = 0) -> list[str]:
        """Rewrite one line as exact half-lines for the phrases it must
        cover, keeping its own words, in order, wherever they fit."""
        rng = random.Random(seed)
        own = [w for w in line.split() if count_syllables(w)]
        pool = lyric_engine._language_pool(mood.language)
        extra: list[str] = []
        for theme in mood.themes + ["connective"]:
            extra.extend(pool.get(theme, []))
        used: list[str] = []
        out: list[str] = []
        for target in targets:
            part: list[str] = []
            room = target
            while own and count_syllables(own[0]) <= room:
                word = own.pop(0)
                part.append(word)
                room -= count_syllables(word)
            if room > 0:
                fresh = [w for w in extra if w not in used + part] or extra
                part.append(lyric_engine.make_line(room, fresh, rng))
            text = " ".join(part)
            used.extend(text.split())
            out.append(text)
        return out


# ==========================================================================
# tune
# ==========================================================================
def alapana_sketch(raaga: Raaga, tonic: int = 60, seed: int = 0,
                   pace: str = "slow") -> list[Note]:
    """A few seconds of free, unmetered phrase that announces the raaga.

    Not a real alapana - a sketch of its outline: settle on Sa, climb the
    arohanam to a jeeva swara and dwell there, come down the avarohanam to a
    resting swara, and close on Sa.  The long notes are the raaga's own
    jeeva and nyasa swaras, so the sketch sounds like this raaga and not
    like a scale.
    """
    rng = random.Random(seed)
    up = [s for s in raaga.arohanam if not s.endswith("+")] or ["S"]
    down = [s for s in raaga.avarohanam if not s.endswith("+")] or ["S"]
    jeeva = {s for s in (raaga.jeeva or []) if s in up[1:]}
    # Climb at least to the middle of the scale, so the sketch travels: a
    # low jeeva swara is dwelt on on the way up rather than turned back at.
    floor = min(len(up) - 1, max(2, (len(up) + 1) // 2))
    high = [s for s in up[floor:] if s in jeeva] or up[floor:]
    peak = rng.choice(high)
    climb = up[:up.index(peak) + 1]
    fall = down[down.index(peak):] if peak in down else down
    # Come to rest on one of the raaga's resting swaras on the way down;
    # the closing Sa is added after, so the fall stops short of it.
    nyasa = [s for s in (raaga.nyasa or []) if s in fall[1:] and s != "S"]
    if nyasa:
        fall = fall[:fall.index(rng.choice(nyasa)) + 1]
    elif fall and fall[-1] == "S":
        fall = fall[:-1]

    long_, mid, short = (2.0, 0.9, 0.45) if pace == "slow" else (1.6, 0.7, 0.36)
    plan: list[tuple[str, float, str]] = [("S", long_, "")]
    for s in climb[1:-1]:
        plan.append((s, mid, "kampita") if s in jeeva else (s, short, ""))
    plan.append((peak, long_ * 1.2, "kampita"))
    for s in fall[1:-1]:
        plan.append((s, short, ""))
    if len(fall) > 1:
        plan.append((fall[-1], mid * 1.5, "kampita"))
    plan.append(("S", long_ * 1.3, ""))

    notes: list[Note] = []
    at = 0.0
    for swara, length, gamaka in plan:
        length *= rng.uniform(0.9, 1.1)          # unmetered, so not even
        notes.append(Note(swara=swara, midi=raaga.midi(swara, tonic),
                          start=round(at, 3), duration=round(length, 3),
                          velocity=84, gamaka=gamaka, section_id="alapana"))
        at += length + 0.06
    return notes


class Tunesmith:
    """Chooses the raaga and writes the phrase.  Never sees the words."""

    name = TUNESMITH

    def __init__(self, raagas: RaagaLibrary | None = None,
                 tonic_midi: int = 60, voice_range: tuple[int, int] = (55, 79)
                 ) -> None:
        self.raagas = raagas or library()
        self.tonic_midi = tonic_midi
        self.voice_range = voice_range

    def propose(self, mood: MoodCard, raaga_name: str = "",
                seed: int = 5, beats_per_cycle: int = 8) -> TuneProposal:
        suggestions = selection.suggest(mood.brief, self.raagas, limit=4)
        alternatives = [(s.name, s.rationale) for s in suggestions]
        chosen_by = TUNESMITH.lower()
        if raaga_name:
            raaga = self.raagas.get(raaga_name)
            if raaga is None:
                raise KeyError(f"I do not know a raaga called {raaga_name}.")
            reason = "the producer chose it"
            known = dict(alternatives)
            if raaga.name in known:
                reason += f" - {known[raaga.name].strip().rstrip('.')}"
            chosen_by = PRODUCER.lower()
        else:
            top = suggestions[0]
            raaga = top.raaga or self.raagas.require(top.name)
            reason = top.rationale.strip().rstrip(".")
        melody = self.phrase(raaga, mood.tempo_bpm, beats_per_cycle, seed)
        return TuneProposal(
            raaga=raaga.name, reason=reason, chosen_by=chosen_by,
            alternatives=alternatives, arohanam=list(raaga.arohanam),
            avarohanam=list(raaga.avarohanam),
            alapana=alapana_sketch(raaga, self.tonic_midi, seed, mood.pace),
            melody=melody, phrase_lines=self._lines(melody), seed=seed)

    def phrase(self, raaga: Raaga, tempo: int, beats_per_cycle: int,
               seed: int) -> MelodyVersion:
        """Two lines, as one Pallavi of a few cycles each."""
        cycle = cycle_seconds(tempo, beats_per_cycle)
        length = cycle * LINES * cycles_per_line(beats_per_cycle)
        sections = [Section(name="Pallavi", kind=SectionKind.PALLAVI,
                            start=0.0, end=length)]
        low, high = self.voice_range
        opts = MelodyOptions(tempo_bpm=int(tempo),
                             beats_per_cycle=beats_per_cycle,
                             tonic_midi=self.tonic_midi, voice_low=low,
                             voice_high=high, seed=seed,
                             duration_target=length, song_type="simple song")
        melody = compose_melody(raaga, opts, sections, version=1,
                                label="Song team phrase")
        return melody

    def recast(self, tune: TuneProposal, beats_per_cycle: int) -> TuneProposal:
        """The same raaga, seed and tempo, set to another cycle."""
        raaga = self.raagas.require(tune.raaga)
        melody = self.phrase(raaga, tune.tempo_bpm, beats_per_cycle, tune.seed)
        return replace(tune, melody=melody, phrase_lines=self._lines(melody))

    @staticmethod
    def _lines(melody: MelodyVersion) -> list[str]:
        cut = melody.duration / LINES if melody.notes else 0.0
        lines: list[list[str]] = [[] for _ in range(LINES)]
        for n in melody.notes:
            idx = min(LINES - 1, int(n.start // cut)) if cut else 0
            lines[idx].append(n.swara)
        return [" ".join(l) for l in lines]


# ==========================================================================
# beat
# ==========================================================================
class TalaKeeper:
    """Chooses the tala for this tune and these words, and lays the beat."""

    name = TALA_KEEPER

    def choose(self, mood: MoodCard, tune: TuneProposal, lyrics: LyricSheet,
               tala_name: str = "") -> tuple[tala_module.Tala, str, str,
                                             list[tuple[str, str]]]:
        alternatives = [(t.name, t.description) for t in tala_module.TALAS]
        if tala_name:
            found = tala_module.find(tala_name)
            if found is None:
                raise KeyError(f"I do not know a tala called {tala_name}.")
            return found, "the producer chose it", PRODUCER.lower(), alternatives
        # The words first: a lullaby sways in three, a chase drives in five.
        read = tala_module.suggest(mood.brief, None)
        default = tala_module.require(tala_module.DEFAULT_TALA)
        if read.tala is not default:
            return (read.tala, read.reason.rstrip("."), TALA_KEEPER.lower(),
                    alternatives)
        # Nothing in the words: read the tune's pace instead.
        tempo = tune.tempo_bpm
        if tempo >= 100:
            return (tala_module.require("Rupaka"),
                    (f"the tune moves briskly ({tempo} bpm) and the mood names "
                    f"no cycle; six beats keeps it light"),
                    TALA_KEEPER.lower(), alternatives)
        return (default,
                (f"the mood names no cycle and the tune is at {tempo} bpm; "
                f"Adi's even eight gives a {mood.pace} line room to breathe"),
                TALA_KEEPER.lower(), alternatives)

    def lay(self, tala: tala_module.Tala, tune: TuneProposal, mood: MoodCard,
            seed: int = 0) -> BeatVersion:
        density = "sparse" if mood.pace == "slow" else tala_module.DEFAULT_DENSITY
        version = BeatVersion(version=1, label="Song team beat",
                              tala=tala.name, tempo_bpm=tune.tempo_bpm,
                              density=density,
                              duration=float(tune.melody.duration),
                              seed=seed,
                              intensity=0.45 if mood.pace == "slow" else 0.6)
        return beat_engine.realise(version)


# ==========================================================================
# the producer's desk
# ==========================================================================
class SongTeam:
    """Runs the simple path: mood -> (lyrics | tune) -> beat -> fit."""

    def __init__(self, raagas: RaagaLibrary | None = None, llm=None,
                 tonic_midi: int = 60,
                 voice_range: tuple[int, int] = (55, 79)) -> None:
        self.listener = MoodListener()
        self.lyricist = Lyricist(llm)
        self.tunesmith = Tunesmith(raagas, tonic_midi, voice_range)
        self.tala_keeper = TalaKeeper()

    def run(self, mood_text: str, language: str = "Tamil", *,
            raaga: str = "", tala: str = "", seed: int | None = None,
            on_event: Callable[[TeamEvent], None] | None = None,
            check: Callable[[], None] | None = None) -> SongDraft:
        """One pass of the team.  ``raaga`` and ``tala`` are the producer's
        direction; empty means "let the agent choose".  ``check`` is called
        between steps so a cancelled job stops promptly."""
        seed = int(seed if seed is not None else time.time()) % 9973 or 1
        events: list[TeamEvent] = []

        def say(agent: str, text: str, to: str = "") -> None:
            event = TeamEvent(agent, text, to)
            events.append(event)
            log.info("%s", event.line())
            if on_event is not None:
                on_event(event)

        def step() -> None:
            if check is not None:
                check()

        say(PRODUCER, f"the mood is: \"{' '.join(mood_text.split())}\"",
            MOOD_LISTENER)
        mood = self.listener.listen(mood_text, language)
        say(MOOD_LISTENER, mood.summary(), f"{LYRICIST} and {TUNESMITH}")
        step()

        # Lyrics and tune at the same time, as Krish asked: each reads only
        # the mood card, so neither waits for the other.
        with ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix="song-team") as pool:
            lyric_job = pool.submit(self.lyricist.write, mood, seed)
            tune_job = pool.submit(self.tunesmith.propose, mood, raaga, seed)
            lyrics = lyric_job.result()
            tune = tune_job.result()
        say(LYRICIST, f"two lines ({lyrics.written_by}): "
            + " / ".join(lyrics.lines), TALA_KEEPER)
        say(TUNESMITH, f"{tune.raaga} ({tune.chosen_by}): {tune.reason}. "
            f"Alapana {tune.alapana_seconds:.1f}s, then two lines at "
            f"{tune.tempo_bpm} bpm.", TALA_KEEPER)
        step()

        chosen, reason, by, alternatives = self.tala_keeper.choose(
            mood, tune, lyrics, tala)
        recast = chosen.aksharas != tune.melody.beats_per_cycle
        if recast:
            tune = self.tunesmith.recast(tune, chosen.aksharas)
            say(TUNESMITH, f"re-set the phrase to {chosen.aksharas} beats a "
                f"cycle for {chosen.name}; same raaga, same notes order",
                TALA_KEEPER)
        beat = self.tala_keeper.lay(chosen, tune, mood, seed)
        proposal = BeatProposal(
            tala=chosen.name, description=chosen.describe(), reason=reason,
            chosen_by=by, alternatives=alternatives,
            tempo_bpm=tune.tempo_bpm, density=beat.density, beat=beat,
            recast_tune=recast)
        say(TALA_KEEPER, f"{chosen.describe()} ({by}): {reason}. "
            f"{len(beat.notes)} strokes, {beat.density}.", PRODUCER)
        step()

        self._fit(lyrics, tune, mood, seed, say)
        say(PRODUCER, "rough picture ready: lyrics + tune + beat")
        return SongDraft(mood=mood, lyrics=lyrics, tune=tune, beat=proposal,
                         events=events, not_done=list(NOT_DONE_YET))

    # -- the fit ------------------------------------------------------------
    def _fit(self, lyrics: LyricSheet, tune: TuneProposal, mood: MoodCard,
             seed: int, say) -> None:
        """Lay the words on the phrase.  A line that cannot be split onto
        its phrases goes back to the lyricist with the exact counts."""
        slots = build_slots(tune.melody)
        groups = self._slots_per_line(slots)
        sung: list[str] = []
        lyrics.revised = []
        for i, line in enumerate(lyrics.lines):
            targets = [s.syllable_count for s in groups[i]] if i < len(groups) else []
            if not targets:
                continue
            parts = split_to_targets(line, targets)
            if parts is None:
                parts = self.lyricist.revise(line, targets, mood, seed + i)
                lyrics.revised.append(i)
                say(LYRICIST, f"line {i + 1} did not fit phrases of "
                    f"{'+'.join(map(str, targets))} syllables; rewrote it as "
                    f"\"{' / '.join(parts)}\"", PRODUCER)
            sung.extend(parts)
        fitted = fit_lines(sung, tune.melody, mood.language)
        lyrics.fitted = fitted
        lyrics.fit_notes = [n for n in fitted.notes.splitlines() if n.strip()]
        # The sung words go onto the notes so anything that plays the phrase
        # (and the JSON hand-off) carries them.
        for line in fitted.lines:
            for syl, idx in zip(line.syllables, line.note_indices):
                if 0 <= idx < len(tune.melody.notes):
                    tune.melody.notes[idx].syllable = syl

    @staticmethod
    def _slots_per_line(slots: list[PhraseSlot]) -> list[list[PhraseSlot]]:
        """Split the phrase's slots between the two lyric lines by time."""
        if not slots:
            return [[] for _ in range(LINES)]
        end = slots[-1].end
        start = slots[0].start
        cut = start + (end - start) / LINES
        first = [s for s in slots if s.start < cut] or slots[:1]
        rest = [s for s in slots if s not in first]
        if not rest and len(first) > 1:
            rest = first[len(first) // 2:]
            first = first[:len(first) // 2]
        return [first, rest]


def split_to_targets(line: str, targets: Sequence[int]) -> list[str] | None:
    """Split a line at word boundaries into parts of exactly these syllable
    counts, or None when the words do not fall that way."""
    words = [w for w in line.split() if w]
    out: list[str] = []
    i = 0
    for target in targets:
        part: list[str] = []
        count = 0
        while i < len(words) and count < target:
            part.append(words[i])
            count += count_syllables(words[i])
            i += 1
        if count != target:
            return None
        out.append(" ".join(part))
    if i != len(words):
        return None
    return out
