"""Regression tests.

Each test here guards a defect that was actually found in this codebase and
fixed. The docstring says what broke, so a future failure is self-explaining.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raagacomposer.audio import dsp
from raagacomposer.core.models import CreativeBrief, VocalDirection
from raagacomposer.core.jobs import JobManager
from raagacomposer.lyrics.fitting import build_slots
from raagacomposer.lyrics.generator import generate_lines
from raagacomposer.music.melody import MelodyOptions, generate, regenerate_section
from raagacomposer.speech.context import ConversationContext
from raagacomposer.speech.intent import interpret
from raagacomposer.speech.timeline_parser import TimeContext, parse
from raagacomposer.voice.profiles import VoiceProfileManager

pytestmark = pytest.mark.regression

SR = 22050
SOURCE = Path(__file__).resolve().parents[2] / "raagacomposer"


# --------------------------------------------------------------------------
# REG-001  source integrity
# --------------------------------------------------------------------------
def test_reg_001_no_control_characters_in_source():
    """A patch once wrote a literal backspace (0x08) into a regex in
    intent.py. The pattern then matched nothing and a whole class of spoken
    instructions silently stopped being understood, invisibly in any editor.
    """
    offenders = []
    for path in SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if any(ord(ch) < 9 or 13 < ord(ch) < 32 for ch in line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"control characters in: {offenders}"


def test_reg_002_the_feel_only_sentence_is_understood():
    """The same defect, seen from the outside: describing a feel with no
    instrument name returned "not understood" instead of a suggestion.
    """
    cmd = interpret("I want this to feel lonely, late at night, but still warm.",
                    TimeContext(duration=180.0))
    assert cmd.intent == "arrange.suggest"
    assert cmd.feel_words


# --------------------------------------------------------------------------
# REG-010  DSP
# --------------------------------------------------------------------------
def test_reg_010_de_esser_reduces_sibilance_instead_of_boosting_it():
    """The de-esser subtracted a phase-shifted copy of the sibilant band from
    the signal, which *added* energy at the sibilant frequency instead of
    ducking it. The band is now extracted with a zero-phase filter and only
    the excess is removed.
    """
    t = np.arange(SR) / SR
    voice = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sibilance = (0.4 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
    mixed = voice + sibilance

    def band(x, low, high):
        spec = np.abs(np.fft.rfft(dsp.as_mono(x)))
        freqs = np.fft.rfftfreq(len(dsp.as_mono(x)), 1 / SR)
        return float(spec[(freqs >= low) & (freqs <= high)].sum())

    out = dsp.de_esser(mixed, SR, freq=6000, threshold_db=-40.0)
    assert band(out, 7000, 8000) < band(mixed, 7000, 8000)
    assert band(out, 250, 350) == pytest.approx(band(mixed, 250, 350), rel=0.25)


def test_reg_011_de_esser_is_transparent_when_nothing_is_sibilant():
    """With the threshold above the signal there must be no change at all."""
    t = np.arange(SR // 2) / SR
    quiet = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = dsp.de_esser(quiet, SR, freq=6000, threshold_db=0.0)
    assert np.allclose(out, quiet, atol=1e-3)


def test_reg_012_soft_clip_never_exceeds_full_scale():
    """soft_clip normalised by tanh(drive), so a signal above full scale came
    out above 1.0 - the opposite of what a clipper is for.
    """
    for drive in (0.5, 1.0, 2.0, 6.0):
        out = dsp.soft_clip(np.linspace(-5, 5, 1000).astype(np.float32), drive)
        assert float(np.abs(out).max()) <= 1.0 + 1e-6, drive


# --------------------------------------------------------------------------
# REG-020  spoken time and level
# --------------------------------------------------------------------------
def test_reg_020_number_words_do_not_match_inside_other_words():
    """The number alternation had no word boundaries, so the "a" inside
    "forward" matched as the number one: "skip forward 20 seconds" moved the
    playhead by one second.
    """
    ctx = TimeContext(duration=240.0, playhead=45.0)
    assert parse("skip forward 20 seconds", ctx).relative == 20.0
    assert parse("go back 10 seconds", ctx).relative == -10.0
    assert parse("skip ahead a minute", ctx).relative == 60.0


def test_reg_021_turn_it_up_with_the_object_in_the_middle():
    """"turn the violin up" was not recognised: the rule only matched
    "turn up" and "turn it up".
    """
    louder = interpret("turn the violin up", TimeContext(duration=100.0))
    softer = interpret("turn the drums down", TimeContext(duration=100.0))
    assert louder.intent == "arrange.level" and louder.value > 1.0
    assert softer.intent == "arrange.level" and softer.value < 1.0


def test_reg_022_the_nth_minute_convention():
    """Specification section 20 phase D names this case explicitly:
    "from the second minute to the third minute" is 01:00-03:00.
    """
    spec = parse("Play from the second minute to the third minute.",
                 TimeContext(duration=400.0))
    assert (spec.start, spec.end) == (60.0, 180.0)


# --------------------------------------------------------------------------
# REG-030  implicit place resolution
# --------------------------------------------------------------------------
def test_reg_030_a_bare_add_is_not_confined_to_one_section():
    """Any arrangement command without a time reference was pinned to the
    section under the playhead. "Add veena" then covered eight seconds of
    prelude instead of the song.
    """
    conversation = ConversationContext(playhead=2.0, duration=200.0)
    cmd = conversation.resolve(interpret("add veena",
                                         conversation.time_context()))
    assert cmd.time is None or cmd.time.start is None


def test_reg_031_replace_is_not_confined_to_the_current_section():
    """Because of the same defect, "replace saxophone with veena" searched
    only the section at the playhead and reported that the saxophone was not
    playing there - when it was playing in the interlude.
    """
    conversation = ConversationContext(playhead=2.0, duration=200.0)
    cmd = conversation.resolve(
        interpret("No, change that saxophone to veena.",
                  conversation.time_context()))
    assert cmd.intent == "arrange.replace"
    assert cmd.instrument == "saxophone"
    assert cmd.target_instrument == "veena"
    assert cmd.time is None or cmd.time.start is None


def test_reg_032_here_still_means_the_playhead():
    """The fix for REG-030 must not break the deictic case."""
    from raagacomposer.core.models import Section, SectionKind
    sections = [Section(name="Pallavi", kind=SectionKind.PALLAVI,
                        start=10.0, end=40.0)]
    conversation = ConversationContext(playhead=20.0, duration=200.0,
                                       sections=sections)
    cmd = conversation.resolve(interpret("add veena here",
                                         conversation.time_context()))
    assert cmd.time.start == 20.0
    assert cmd.time.end == 40.0


# --------------------------------------------------------------------------
# REG-040  voice profile lookup
# --------------------------------------------------------------------------
def test_reg_040_asking_for_a_male_voice_does_not_return_a_female_one(tmp_path):
    """by_name matched substrings, and "male" is a substring of "female", so
    asking for the male singer selected "Female - Warm".
    """
    manager = VoiceProfileManager(tmp_path / "voices.json")
    assert manager.by_name("male").gender == "male"
    assert manager.by_name("female").gender == "female"
    assert manager.default("male").gender == "male"


# --------------------------------------------------------------------------
# REG-050  jobs
# --------------------------------------------------------------------------
def test_reg_050_an_uninterruptible_job_lands_stale_not_cancelled():
    """Specification section 17 distinguishes the two: a provider call that
    cannot be cancelled is allowed to finish, and its output is then rejected
    as stale. The manager marked it "cancelled" and set its cancel flag, which
    misreported what actually happened.
    """
    import threading
    import time

    jobs = JobManager(max_workers=3)
    delivered = []
    release = threading.Event()

    job = jobs.submit("test", "same", lambda ctx: release.wait(3.0) or "old",
                      on_done=delivered.append, cancellable=False)
    time.sleep(0.05)
    jobs.submit("test", "same", lambda ctx: "new", on_done=delivered.append)

    deadline = time.time() + 5
    while time.time() < deadline and "new" not in delivered:
        jobs.drain()
        time.sleep(0.01)
    release.set()
    deadline = time.time() + 5
    while time.time() < deadline and job.status not in ("stale", "done"):
        jobs.drain()
        time.sleep(0.01)

    assert job.status == "stale"
    assert not job.cancel_event.is_set()
    assert "old" not in delivered
    jobs.shutdown()


# --------------------------------------------------------------------------
# REG-060  musical invariants
# --------------------------------------------------------------------------
def test_reg_060_regenerating_a_section_leaves_the_others_bit_identical(keeravani):
    """The core non-destructive-editing promise (specification section 16)."""
    opts = MelodyOptions(tempo_bpm=72, seed=5, duration_target=120)
    melody = generate(keeravani, opts)
    target = melody.sections[1]
    before = [(n.start, n.midi, n.swara, n.gamaka) for n in melody.notes
              if n.section_id != target.id]
    fresh = regenerate_section(melody, keeravani, target.id, opts, 2)
    after = [(n.start, n.midi, n.swara, n.gamaka) for n in fresh.notes
             if n.section_id != target.id]
    assert before == after


def test_reg_061_a_section_never_repeats_the_same_line_twice_running(keeravani):
    """The pallavi-repeat rule keyed only on the section name, so consecutive
    phrases inside one section were handed the identical line.
    """
    melody = generate(keeravani, MelodyOptions(tempo_bpm=60, seed=21,
                                               duration_target=150))
    slots = build_slots(melody)
    lines = generate_lines(slots, CreativeBrief(language="Tamil"), seed=9)
    for i in range(1, len(slots)):
        if slots[i].section_id == slots[i - 1].section_id:
            assert lines[i] != lines[i - 1], slots[i].section_name


# --------------------------------------------------------------------------
# REG-070  desktop layout
# --------------------------------------------------------------------------
@pytest.mark.ui
def test_reg_070_the_window_does_not_demand_a_huge_width():
    """The arrangement controls and the timeline's own width propagated out
    through the scroll area, forcing the window to about 2700 px - unusable
    on an ordinary screen.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from raagacomposer.app import AppController
    from raagacomposer.core.settings import Settings
    from raagacomposer.ui import theme
    from raagacomposer.ui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.STYLESHEET)
    settings = Settings.load()
    controller = AppController(settings)
    window = MainWindow(controller)
    try:
        window.resize(1400, 900)
        window.show()
        application.processEvents()
        assert window.minimumSizeHint().width() <= 1280
        assert window.width() == 1400
    finally:
        window._timer.stop()
        controller.close()


# --------------------------------------------------------------------------
# REG-080  playhead
# --------------------------------------------------------------------------
def test_reg_080_the_playhead_moves_before_anything_is_rendered(app):
    """The playhead lived only in the audio engine, which clamps to the loaded
    buffer. Before the first render that buffer is empty, so every "here"
    resolved to 0:00 no matter where the creator clicked.
    """
    app.new_project("Playhead")
    app.project.brief.duration_target = 120.0
    app.seek(42.0)
    assert app.playhead == pytest.approx(42.0)
    app._sync_context()
    assert app.context.playhead == pytest.approx(42.0)


def test_reg_081_an_unavailable_instrument_is_reported_even_when_parsed(app):
    """"Add a theremin here" parsed as a valid add command with no instrument,
    so the unavailable-instrument report never ran and the creator was told
    only "which instrument would you like?".
    """
    app.new_project("Unavailable")
    before = len(app.project.errors)
    app.handle_utterance("Add a theremin here.")
    assert len(app.project.errors) > before
    message = app.project.errors[-1].message
    assert "theremin" in message and "Closest available" in message
