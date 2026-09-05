"""Ascent and descent grammar in the finished line.

Found by measuring the composer on a raaga that is not Keeravani.  The
generator walks in scale-degree space and takes each next note from the
ascending or descending ladder as the phrase requires - but octaves are then
placed to fit a register, and phrases are joined into sections, and neither
of those knows which way the line is travelling.  Either can leave a note
unchanged while reversing the motion into it, which in a raaga whose
arohanam and avarohanam differ turns a legal descending step into an
ascending leap onto a note the arohanam forbids.

Measured before the fix, over 20 seeds each: Abheri 13 bad moves in 735,
Kambhoji 4, Bhairavi 4, Sindhu Bhairavi 56 in 16 of 20 seeds.  None of it
was visible on Keeravani, Kalyani or Mohanam, whose two ladders hold the
same swaras, and Keeravani was the only raaga ever measured.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.music import melody as melody_engine
from raagacomposer.music.melody import MelodyOptions, enforce_direction
from raagacomposer.music.structure import plan_sections
from raagacomposer.raaga.library import RaagaLibrary, parse_swara

pytestmark = pytest.mark.regression

NO_USER_FILE = Path("no-such-user-raagas.json")

#: Every curated raaga whose arohanam and avarohanam differ.  These are the
#: only ones that can break the rule, and so the only ones worth measuring.
ASYMMETRIC = ("Abheri", "Kambhoji", "Bhairavi", "Sindhu Bhairavi")
SYMMETRIC = ("Keeravani", "Kalyani", "Mohanam")


@pytest.fixture(scope="module")
def lib() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE)


def _tune(raaga, seed: int):
    opts = MelodyOptions(seed=seed, tempo_bpm=72, duration_target=45.0)
    sections = plan_sections(opts.duration_target, opts.tempo_bpm,
                             opts.beats_per_cycle, opts.song_type)
    return melody_engine.generate(raaga, opts, sections).notes


def _illegal_moves(raaga, notes) -> list:
    ascending, descending = set(raaga.ascending), set(raaga.descending)
    bad = []
    for i in range(1, len(notes)):
        step = notes[i].midi - notes[i - 1].midi
        base = parse_swara(notes[i].swara)[0]
        if step > 0 and base not in ascending:
            bad.append(f"{notes[i - 1].swara}->{notes[i].swara}")
        elif step < 0 and base not in descending:
            bad.append(f"{notes[i - 1].swara}->{notes[i].swara}")
    return bad


@pytest.mark.parametrize("name", ASYMMETRIC)
def test_an_asymmetric_raaga_never_rises_onto_a_descent_only_note(lib, name):
    raaga = lib.require(name)
    for seed in range(1, 21):
        bad = _illegal_moves(raaga, _tune(raaga, seed))
        assert not bad, f"{name} seed {seed}: {bad[:4]}"


@pytest.mark.parametrize("name", SYMMETRIC)
def test_a_symmetric_raaga_is_left_exactly_as_the_walk_produced_it(lib, name):
    """Both ladders hold the same swaras, so nothing can be illegal and the
    repair must not touch a single note - which is what keeps the golden
    melodies byte-identical."""
    raaga = lib.require(name)
    for seed in (3, 7, 11):
        notes = _tune(raaga, seed)
        before = [(n.swara, n.midi) for n in notes]
        assert enforce_direction(raaga, notes, 60) == 0
        assert [(n.swara, n.midi) for n in notes] == before


def test_the_repair_keeps_the_direction_of_travel(lib, monkeypatch):
    """A repaired note still moves the way the line was moving.  It is the
    note that changes, never the shape: every pair that rose before the
    repair still rises after it, and every pair that fell still falls."""
    raaga = lib.require("Sindhu Bhairavi")
    for seed in range(1, 11):
        monkeypatch.setattr(melody_engine, "enforce_direction",
                            lambda *a, **k: 0)
        unrepaired = [(n.midi) for n in _tune(raaga, seed)]
        monkeypatch.undo()
        repaired = [(n.midi) for n in _tune(raaga, seed)]

        assert len(unrepaired) == len(repaired)
        for i in range(1, len(repaired)):
            was = unrepaired[i] - unrepaired[i - 1]
            now = repaired[i] - repaired[i - 1]
            if was and now:
                assert (was > 0) == (now > 0), (
                    f"seed {seed} note {i}: motion reversed by the repair")


def test_a_repair_is_never_a_wild_leap(lib):
    """Nothing further than a fifth is substituted: past that the repair
    would be a worse artefact than the fault it fixes."""
    raaga = lib.require("Sindhu Bhairavi")
    for seed in range(1, 11):
        notes = _tune(raaga, seed)
        for i in range(1, len(notes)):
            assert abs(notes[i].midi - notes[i - 1].midi) <= 24, \
                f"seed {seed} note {i}: {notes[i - 1].swara}->{notes[i].swara}"
