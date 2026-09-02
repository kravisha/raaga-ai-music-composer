"""Defects found by running a real local model against the lyric engine.

Each test names something that actually happened here with llama3.2:3b.
"""
from __future__ import annotations

import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.lyrics import generator as lyric_generator
from raagacomposer.lyrics.fitting import build_slots, count_syllables

pytestmark = pytest.mark.regression


class ScriptIgnoringLLM:
    """Answers in Tamil script though the prompt asked for transliteration."""

    name = "native-script"
    available = True

    def __init__(self, lines):
        self.lines = lines

    def write_lyrics(self, slots, brief):
        return list(self.lines)[:len(slots)]


def test_reg_097_a_line_in_a_script_the_singer_cannot_read_is_replaced(
        short_melody, brief):
    """llama3.2:3b answered in Tamil script. The syllable engine counts only
    Roman letters, so every line scored zero syllables and the synthesiser
    had nothing to sing - yet the lines were accepted and fitted anyway."""
    slots = build_slots(short_melody)
    llm = ScriptIgnoringLLM(["மாலையில்"] * len(slots))

    lyrics = lyric_generator.generate(short_melody, brief, version=1, seed=5,
                                      llm=llm)

    assert len(lyrics.lines) == len(slots)
    for line in lyrics.lines:
        assert count_syllables(line.text) > 0, f"unsingable: {line.text!r}"


def test_reg_097_a_usable_line_beside_an_unusable_one_keeps_its_slot(
        short_melody, brief):
    """Replacement is positional: a good line must not slide onto the slot
    that belonged to the line before it."""
    slots = build_slots(short_melody)
    good = "kaathiruppen vaa"
    drafted = ["தமிழ்"] * len(slots)
    drafted[1] = good
    lyrics = lyric_generator.generate(short_melody, brief, version=1, seed=5,
                                      llm=ScriptIgnoringLLM(drafted))

    assert lyrics.lines[1].text.startswith("kaathiruppen")
    for line in lyrics.lines:
        assert count_syllables(line.text) > 0


def test_reg_097_lines_that_are_already_singable_are_left_alone(short_melody,
                                                               brief):
    slots = build_slots(short_melody)
    drafted = [f"nilavu oru {i}" for i in range(len(slots))]
    lyrics = lyric_generator.generate(short_melody, brief, version=1, seed=5,
                                      llm=ScriptIgnoringLLM(drafted))
    assert all("nilavu" in line.text for line in lyrics.lines)
