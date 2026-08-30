"""Golden-file regression tests.

These pin the *output* of the deterministic engines. A change here is not
necessarily a bug -- but it is never accidental, and it has to be looked at and
approved rather than slipping through unnoticed.

To approve a deliberate change::

    RAAGA_UPDATE_GOLDEN=1 .venv\\Scripts\\python.exe -m pytest tests/regression

and review the diff to tests/golden/ before committing it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from raagacomposer.core.models import CreativeBrief, Section, SectionKind
from raagacomposer.lyrics.fitting import build_slots, count_syllables
from raagacomposer.lyrics.generator import generate_lines
from raagacomposer.music.melody import MelodyOptions, generate
from raagacomposer.music.structure import plan_sections
from raagacomposer.speech.intent import interpret
from raagacomposer.speech.timeline_parser import TimeContext, parse

pytestmark = pytest.mark.regression

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
UPDATE = os.environ.get("RAAGA_UPDATE_GOLDEN") == "1"


def check(name: str, produced: Any) -> None:
    """Compare against the stored golden file, or write it when approving."""
    path = GOLDEN / f"{name}.json"
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(produced, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        if not UPDATE:
            pytest.skip(f"created missing golden file {path.name}")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert produced == expected, (
        f"{name} changed. If this was intended, re-run with "
        f"RAAGA_UPDATE_GOLDEN=1 and review the diff to {path.name}.")


def _sections():
    return [
        Section(name="Prelude", kind=SectionKind.PRELUDE, start=0, end=12),
        Section(name="Pallavi", kind=SectionKind.PALLAVI, start=12, end=40),
        Section(name="Interlude 1", kind=SectionKind.INTERLUDE, start=40, end=52),
        Section(name="Charanam 1", kind=SectionKind.CHARANAM, start=52, end=80),
        Section(name="Outro", kind=SectionKind.OUTRO, start=80, end=92),
    ]


# --------------------------------------------------------------------------
# melody
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raaga_name,seed", [("Keeravani", 7), ("Mohanam", 3),
                                             ("Kalyani", 11)])
def test_golden_melody(raagas, raaga_name, seed):
    raaga = raagas.require(raaga_name)
    melody = generate(raaga, MelodyOptions(tempo_bpm=72, seed=seed,
                                           duration_target=120,
                                           tonic_midi=60, voice_low=52,
                                           voice_high=79))
    produced = {
        "raaga": melody.raaga,
        "tempo": melody.tempo_bpm,
        "sections": [[s.name, round(s.start, 3), round(s.end, 3)]
                     for s in melody.sections],
        "notes": [[n.swara, n.midi, round(n.start, 3), round(n.duration, 3),
                   n.velocity, n.gamaka] for n in melody.notes],
    }
    check(f"melody_{raaga_name.lower()}_seed{seed}", produced)


def test_golden_structure():
    produced = {
        f"{song_type}_{int(target)}": [[s.name, round(s.start, 3), round(s.end, 3)]
                                       for s in plan_sections(target, 72, 8,
                                                              song_type)]
        for song_type in ("film song", "devotional", "simple")
        for target in (90.0, 150.0, 240.0)
    }
    check("structure_plans", produced)


# --------------------------------------------------------------------------
# spoken language
# --------------------------------------------------------------------------
TIMELINE_PHRASES = [
    "Play the first minute.",
    "Play from the second minute to the third minute.",
    "Play the end.",
    "Play the last 30 seconds.",
    "Play from the chorus.",
    "Play this section again.",
    "Start five seconds before this point.",
    "Go back 10 seconds.",
    "play from 1:20 to 2:05",
    "play the whole song",
    "use only piano for the first 15 seconds",
    "add mridangam after the chorus",
    "bring strings before the charanam",
    "play the outro",
    "skip forward 20 seconds",
]


def test_golden_timeline_parsing():
    ctx = TimeContext(duration=240.0, playhead=45.0, sections=_sections())
    produced = {}
    for phrase in TIMELINE_PHRASES:
        spec = parse(phrase, ctx)
        produced[phrase] = None if spec is None else {
            "start": None if spec.start is None else round(spec.start, 3),
            "end": None if spec.end is None else round(spec.end, 3),
            "relative": spec.relative,
            "source": spec.source,
        }
    check("timeline_parsing", produced)


INTENT_PHRASES = [
    "Add veena here.",
    "Use saxophone for this interlude.",
    "Bring strings after this line.",
    "Take the drums out here.",
    "Use only piano for the first 15 seconds.",
    "Add mridangam after the chorus.",
    "Replace violin with veena.",
    "Make this part lighter.",
    "Give me another instrument that fits this feel.",
    "I want this to feel lonely, late at night, but still warm.",
    "Give me the song without instruments.",
    "No, change that violin to saxophone.",
    "Play from the second minute to the third minute.",
    "Set the tempo to 96 bpm",
    "Use raaga Kalyani.",
    "Lock the pallavi.",
    "turn the violin up",
    "Mix the song.",
    "Undo.",
]


def test_golden_intent_classification():
    ctx = TimeContext(duration=240.0, playhead=45.0, sections=_sections())
    produced = {}
    for phrase in INTENT_PHRASES:
        cmd = interpret(phrase, ctx, last_instrument="violin")
        produced[phrase] = {
            "intent": cmd.intent,
            "instrument": cmd.instrument,
            "target": cmd.target_instrument,
            "raaga": cmd.raaga,
            "value": cmd.value,
            "time": None if cmd.time is None else [
                None if cmd.time.start is None else round(cmd.time.start, 3),
                None if cmd.time.end is None else round(cmd.time.end, 3),
            ],
        }
    check("intent_classification", produced)


# --------------------------------------------------------------------------
# lyrics
# --------------------------------------------------------------------------
def test_golden_lyrics(raagas):
    raaga = raagas.require("Charukesi")
    melody = generate(raaga, MelodyOptions(tempo_bpm=68, seed=7,
                                           duration_target=120))
    slots = build_slots(melody)
    produced = {}
    for language in ("Tamil", "Hindi", "Telugu", "English"):
        lines = generate_lines(slots, CreativeBrief(language=language,
                                                    mood="longing"), seed=5)
        produced[language] = lines
        # the invariant that matters, checked independently of the golden text
        for line, slot in zip(lines, slots):
            assert count_syllables(line) == slot.syllable_count
    check("lyrics_lines", produced)
