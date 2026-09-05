"""Unit tests: timeline parsing, intent interpretation and conversation state."""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.core.models import Section, SectionKind
from raagacomposer.speech.context import ConversationContext
from raagacomposer.speech.intent import (Command, describe, interpret,
                                         unavailable_instrument)
from raagacomposer.speech.stt import (TypedSTT, Transcript, _resample,
                                      build_adapter)
from raagacomposer.speech.timeline_parser import (TimeContext, TimeSpec,
                                                  describe as describe_time,
                                                  parse)

pytestmark = pytest.mark.unit


def sections():
    return [
        Section(name="Prelude", kind=SectionKind.PRELUDE, start=0, end=12),
        Section(name="Pallavi", kind=SectionKind.PALLAVI, start=12, end=40),
        Section(name="Interlude 1", kind=SectionKind.INTERLUDE, start=40, end=52),
        Section(name="Charanam 1", kind=SectionKind.CHARANAM, start=52, end=80),
        Section(name="Pallavi 2", kind=SectionKind.PALLAVI, start=80, end=104),
        Section(name="Outro", kind=SectionKind.OUTRO, start=104, end=116),
    ]


def ctx(duration=240.0, playhead=45.0, selection=None):
    return TimeContext(duration=duration, playhead=playhead, selection=selection,
                       sections=sections())


# --------------------------------------------------------------------------
# timeline: ranges
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,start,end", [
    ("Play the first minute.", 0.0, 60.0),
    ("Play from the second minute to the third minute.", 60.0, 180.0),
    ("play from the first minute to the second minute", 0.0, 120.0),
    ("Play the last 30 seconds.", 210.0, 240.0),
    ("play the first 15 seconds", 0.0, 15.0),
    ("play from 1:20 to 2:05", 80.0, 125.0),
    ("play between 0:30 and 0:45", 30.0, 45.0),
    ("play the whole song", 0.0, 240.0),
])
def test_ranges(text, start, end):
    spec = parse(text, ctx())
    assert spec is not None, text
    assert spec.start == pytest.approx(start), text
    assert spec.end == pytest.approx(end), text


def test_the_nth_minute_convention_is_the_one_the_spec_asks_for():
    """'second minute to third minute' must mean 01:00-03:00."""
    spec = parse("Play from the second minute to the third minute.", ctx())
    assert (spec.start, spec.end) == (60.0, 180.0)


def test_ranges_are_clamped_to_the_song():
    spec = parse("Play from the second minute to the third minute.",
                 ctx(duration=94.0))
    assert spec.start == 60.0
    assert spec.end == pytest.approx(94.0)


def test_the_end_is_the_tail_of_the_song():
    spec = parse("Play the end.", ctx())
    assert spec.end == pytest.approx(240.0)
    assert 0 < spec.start < spec.end


def test_the_beginning_starts_at_zero():
    spec = parse("play the beginning", ctx())
    assert spec.start == 0.0


# --------------------------------------------------------------------------
# timeline: points, sections, relatives
# --------------------------------------------------------------------------
def test_named_sections_resolve():
    assert parse("Play from the chorus.", ctx()).start == 12.0
    assert parse("play the interlude", ctx()).start == 40.0
    assert parse("play the outro", ctx()).start == 104.0
    assert parse("play the second pallavi", ctx()).start == 80.0


def test_before_and_after_a_section():
    assert parse("Add mridangam after the chorus.", ctx()).start == 40.0
    assert parse("bring strings before the charanam", ctx()).start == 52.0


def test_playhead_relative_references():
    assert parse("Start five seconds before this point.", ctx()).start == 40.0
    assert parse("ten seconds after this point", ctx()).start == 55.0
    assert parse("add veena here", ctx()).start == 45.0


def test_relative_moves():
    assert parse("Go back 10 seconds.", ctx()).relative == -10.0
    assert parse("skip forward 20 seconds", ctx()).relative == 20.0
    assert parse("rewind two minutes", ctx()).relative == -120.0


def test_this_part_uses_the_selection_then_the_section():
    with_selection = parse("regenerate this part", ctx(selection=(30.0, 50.0)))
    assert (with_selection.start, with_selection.end) == (30.0, 50.0)
    without = parse("play this section again", ctx(playhead=45.0))
    assert (without.start, without.end) == (40.0, 52.0)


def test_absolute_and_clock_stamps():
    assert parse("play from 45 seconds", ctx()).start == 45.0
    assert parse("jump to 2:30", ctx()).start == 150.0
    assert parse("play for 20 seconds", ctx(playhead=10.0)).end == 30.0


def test_no_time_reference_returns_none():
    assert parse("make it sound warmer", ctx()) is None
    assert parse("", ctx()) is None


def test_time_spec_helpers():
    assert TimeSpec(start=1.0, end=2.0).is_range
    assert TimeSpec(start=1.0).is_point
    assert describe_time(None) == "no time reference"
    assert "-" in describe_time(TimeSpec(start=0.0, end=60.0))


# --------------------------------------------------------------------------
# intent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,intent,instrument", [
    ("Add veena here.", "arrange.add", "veena"),
    ("Use saxophone for this interlude.", "arrange.add", "saxophone"),
    ("Bring strings after this line.", "arrange.add", "strings"),
    ("Take the drums out here.", "arrange.remove", "drum_kit"),
    ("Use only piano for the first 15 seconds.", "arrange.add", "piano"),
    ("Add mridangam after the chorus.", "arrange.add", "mridangam"),
    ("Replace violin with veena.", "arrange.replace", "violin"),
    ("Make this part lighter.", "arrange.level", ""),
    ("Give me another instrument that fits this feel.", "arrange.suggest", ""),
    ("Give me the song without instruments.", "voice.vocal_only", ""),
    ("Play the first minute.", "transport.play", ""),
    ("Stop.", "transport.stop", ""),
    ("Pause.", "transport.pause", ""),
    ("Continue.", "transport.resume", ""),
    ("Go back 10 seconds.", "transport.seek", ""),
    ("Use raaga Kalyani.", "raaga.set", ""),
    ("Set the tempo to 96 bpm", "tune.tempo", ""),
    ("Lock the pallavi.", "region.lock", ""),
    ("Undo.", "project.undo", ""),
    ("Save.", "project.save", ""),
    ("Mix the song.", "mix.full", ""),
    ("Give me the instrumental.", "mix.instrumental", ""),
])
def test_spec_example_utterances(text, intent, instrument):
    cmd = interpret(text, ctx())
    assert cmd.intent == intent, f"{text!r} -> {cmd.intent}"
    if instrument:
        assert cmd.instrument == instrument


def test_replace_extracts_both_sides():
    cmd = interpret("No, change that violin to saxophone.", ctx())
    assert cmd.intent == "arrange.replace"
    assert cmd.instrument == "violin"
    assert cmd.target_instrument == "saxophone"


def test_a_feel_with_no_instrument_becomes_a_suggestion():
    cmd = interpret("I want this to feel lonely, late at night, but still warm.",
                    ctx())
    assert cmd.intent == "arrange.suggest"
    assert {"lonely", "night", "warm"} <= set(cmd.feel_words)


def test_instrument_and_time_are_extracted_together():
    cmd = interpret("Use saxophone for this interlude.", ctx(playhead=45.0))
    assert cmd.instrument == "saxophone"
    assert (cmd.time.start, cmd.time.end) == (40.0, 52.0)
    assert cmd.section_id


def test_raaga_names_are_recognised():
    cmd = interpret("switch to Kalyani", ctx())
    assert cmd.raaga == "Kalyani"
    assert cmd.intent == "raaga.set"


def test_tempo_slots():
    assert interpret("set the tempo to 108 bpm", ctx()).value == 108.0
    faster = interpret("make it faster", ctx())
    assert faster.intent == "tune.tempo" and faster.value and faster.value > 1


def test_level_direction():
    softer = interpret("make the violin softer", ctx())
    louder = interpret("turn the violin up", ctx())
    assert softer.value < 1.0 < louder.value


def test_unknown_utterance_is_reported_as_unknown():
    cmd = interpret("the quick brown fox", ctx())
    assert cmd.intent == "unknown"
    assert cmd.interpretation == "Not understood"


def test_unavailable_instrument_is_named_with_alternatives():
    found = unavailable_instrument("add a theremin here")
    assert found is not None
    phrase, alternatives = found
    assert phrase == "theremin"
    assert alternatives and all(isinstance(a, str) for a in alternatives)
    assert unavailable_instrument("add veena here") is None


def test_target_key_groups_work_for_supersession():
    a = interpret("Add veena here.", ctx())
    b = interpret("Add veena here.", ctx())
    assert a.target_key() == b.target_key()
    assert interpret("Play the end.", ctx()).target_key() == "playback"
    assert interpret("Write the lyrics.", ctx()).target_key() == "lyrics"


def test_describe_is_human_readable():
    cmd = interpret("Use saxophone for this interlude.", ctx())
    assert "Saxophone" in describe(cmd)


# --------------------------------------------------------------------------
# conversation context
# --------------------------------------------------------------------------
def test_context_fills_in_the_last_instrument():
    conversation = ConversationContext(playhead=45.0, duration=240.0,
                                       sections=sections(),
                                       last_instrument="violin")
    cmd = interpret("make that softer", conversation.time_context(),
                    last_instrument="violin")
    cmd = conversation.resolve(cmd)
    assert cmd.instrument == "violin"


def test_a_bare_add_means_the_whole_song():
    """No place named: the controller spans the song rather than guessing."""
    conversation = ConversationContext(playhead=45.0, duration=240.0,
                                       sections=sections())
    cmd = conversation.resolve(interpret("add veena", conversation.time_context()))
    assert cmd.time is None or cmd.time.start is None


def test_here_means_the_playhead_to_the_end_of_that_section():
    conversation = ConversationContext(playhead=45.0, duration=240.0,
                                       sections=sections())
    cmd = conversation.resolve(interpret("add veena here",
                                         conversation.time_context()))
    assert cmd.time.start == 45.0
    assert cmd.time.end == 52.0


def test_remove_and_replace_do_not_invent_a_range():
    """They act wherever that instrument already plays."""
    conversation = ConversationContext(playhead=45.0, duration=240.0,
                                       sections=sections())
    for text in ("replace violin with veena", "remove the violin"):
        cmd = conversation.resolve(interpret(text, conversation.time_context()))
        assert cmd.time is None or cmd.time.start is None, text


def test_context_prefers_an_explicit_selection():
    conversation = ConversationContext(playhead=45.0, duration=240.0,
                                       sections=sections(),
                                       selection=(10.0, 20.0))
    cmd = conversation.resolve(interpret("add veena", conversation.time_context()))
    assert (cmd.time.start, cmd.time.end) == (10.0, 20.0)


def test_open_ended_range_runs_to_the_end_of_the_section():
    conversation = ConversationContext(playhead=0.0, duration=240.0,
                                       sections=sections())
    cmd = interpret("add veena from 45 seconds", conversation.time_context())
    cmd = conversation.resolve(cmd)
    assert cmd.time.start == 45.0
    assert cmd.time.end == 52.0


def test_remember_updates_the_last_mentioned_things():
    conversation = ConversationContext(duration=240.0, sections=sections())
    cmd = interpret("Replace violin with veena.", conversation.time_context())
    conversation.remember(cmd)
    assert conversation.last_instrument == "veena"
    assert conversation.last_intent == "arrange.replace"


def test_turns_are_recorded_and_capped():
    conversation = ConversationContext()
    turn = conversation.add_turn("hello", intent="unknown")
    conversation.update_status(turn.id, "applied", ["track:1"])
    assert conversation.turns[-1].status == "applied"
    assert conversation.turns[-1].targets == ["track:1"]
    for i in range(600):
        conversation.add_turn(f"line {i}")
    assert len(conversation.turns) == 500
    assert "You:" in conversation.transcript()


# --------------------------------------------------------------------------
# speech adapters
# --------------------------------------------------------------------------
def test_typed_adapter_is_always_available():
    adapter = TypedSTT()
    assert adapter.available
    assert "typed" in adapter.status()


def test_whisper_does_not_load_its_model_to_say_it_is_installed():
    """``build_adapter`` constructs every candidate while the application
    is starting, so a constructor that loads a model charges fifteen
    seconds of startup to creators who never press the microphone."""
    from raagacomposer.speech.stt import WhisperSTT

    stt = WhisperSTT("tiny")
    # Installed or not, the constructor must not have loaded anything.
    assert stt._model is None
    if stt.available:
        assert "loads on first use" in stt.status()


def test_build_adapter_falls_back_to_typed(settings):
    settings.stt_provider = "none"
    assert build_adapter(settings).name == "typed"


def test_resampling_changes_length_not_shape():
    audio = np.sin(np.linspace(0, 20, 16000)).astype(np.float32)
    out = _resample(audio, 16000, 8000)
    assert len(out) == 8000
    assert out.ndim == 1


def test_transcript_dataclass_defaults():
    t = Transcript("play the end")
    assert t.final and t.confidence == 1.0


# --------------------------------------------------------------------------
# what the microphone is doing, visibly
# --------------------------------------------------------------------------
def test_the_microphone_says_what_it_is_doing():
    """Voice gave no sign of its state: you spoke, and either something
    happened or nothing did.  Section 15 asks for the phases to be visible."""
    from raagacomposer.speech.capture import CaptureState, VoiceInputManager

    manager = VoiceInputManager.__new__(VoiceInputManager)
    manager.state = CaptureState(backend="typed")
    manager.adapter = TypedSTT()

    assert "off" in manager.status_text().lower()

    manager.state.listening = True
    manager.state.phase = "listening"
    assert "listening" in manager.status_text().lower()

    manager.state.phase = "hearing"
    assert "hearing" in manager.status_text().lower()

    manager.state.phase = "thinking"
    assert manager.status_text() == "Working out what you said"

    manager.state.phase = "done"
    manager.state.heard = "generate a tune"
    assert "generate a tune" in manager.status_text()

    manager.state.error = "Microphone error: no such device"
    assert manager.status_text() == "Microphone error: no such device"
