"""The funky groove (raagacomposer/studio/groove.py): Krish's clip's feel, our own hook."""
from __future__ import annotations

import pytest

from raagacomposer.studio import groove as G

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def built():
    return G.build()


def test_the_groove_runs_at_the_clips_tempo_and_two_bars_are_one_adi_cycle(built):
    assert built.bpm == 140
    beat = 60 / built.bpm
    assert abs(G.STEPS * built.step_seconds - 4 * beat) < 1e-9
    assert abs(built.seconds - built.bars * 4 * beat) < 1e-6


def test_kick_on_one_and_three_clap_on_two_and_four(built):
    bar = G.STEPS * built.step_seconds
    first = [n for n in built.parts["drum_kit"] if n.start < bar - 1e-6]
    kicks = sorted(round(n.start / built.step_seconds) for n in first if n.midi % 3 == 0)
    claps = sorted(round(n.start / built.step_seconds) for n in first if n.midi % 3 == 1 and n.velocity > 100)
    assert kicks[:2] == [0, 8]
    assert claps == [4, 12]


def test_every_hook_and_bass_note_is_in_shanmukhapriya(built, raagas):
    raaga = raagas.require("Shanmukhapriya")
    for part in ("electric_piano", "bass", "synth_pad"):
        for n in built.parts[part]:
            assert raaga.is_allowed(n.swara), (part, n.swara)


def test_the_hook_waits_for_the_intro_and_the_kanjira_joins_the_second_time(built):
    bar = G.STEPS * built.step_seconds
    hook = built.parts["electric_piano"]
    assert min(n.start for n in hook) >= built.intro_bars * bar - 1e-6
    second_time = (built.intro_bars + len(G.HOOK)) * bar
    assert built.parts["kanjira"] and min(n.start for n in built.parts["kanjira"]) >= second_time - 1e-6


def test_it_renders_as_stereo_audio_of_the_right_length():
    audio, g = G.render(sr=22050)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert abs(len(audio) / 22050 - (g.seconds + 1.5)) < 0.05
    assert float(abs(audio).max()) < 1.0


def test_the_entrance_is_ten_seconds_of_silence_then_about_ten_seconds_to_win_the_crowd():
    g, parts, seconds = G.build_show(10.0)
    first = min(n.start for notes in parts.values() for n in notes)
    assert abs(first - 10.0) < 1e-6, "nothing sounds before the entrance"
    entrance_s = G.ENTRANCE_BARS * G.STEPS * g.step_seconds
    assert 9.5 < entrance_s < 11.0
    groove_start = 10.0 + entrance_s
    hook = [n for n in parts["electric_piano"] if n.start >= groove_start - 1e-6]
    assert hook, "the groove follows the entrance"
    assert abs(seconds - (groove_start + g.seconds)) < 1e-6


def test_every_note_of_the_entrance_is_in_the_raaga(raagas):
    raaga = raagas.require("Shanmukhapriya")
    _, parts, _ = G.build_show(10.0)
    for part in ("electric_piano", "bass"):
        for n in parts[part]:
            assert raaga.is_allowed(n.swara), (part, n.swara)
