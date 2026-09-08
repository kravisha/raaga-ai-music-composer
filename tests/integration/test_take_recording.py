"""Integration: recording a take of your own - Record, Stop, Cancel.

Real controller, an injected input stream in place of the microphone.
No device is opened: the fake stream is fed by the test, block by block,
through the same callback sounddevice would call.  What is checked is the
lifecycle (the input is held only between Record and Stop, and released
on Cancel, on a song switch and on close), the shape of what is kept (a
WAV of exactly the audio delivered, at the app's rate, with the place in
the song it was made against), and that nothing else in the song moves.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.core.models import SectionKind
from raagacomposer.voice.recorder import NO_BACKEND, TakeRecorder

pytestmark = pytest.mark.integration


class FakeStream:
    """What sounddevice.InputStream looks like from the recorder's side."""

    def __init__(self, samplerate, channels, blocksize, device, callback):
        self.samplerate, self.channels, self.blocksize = samplerate, channels, blocksize
        self.device, self.callback = device, callback
        self.started = False
        self.stopped = False
        self.closed = False
        self.delivered = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, seconds: float, level: float = 0.3, seed: int = 1) -> int:
        """Deliver ``seconds`` of noise in blocks, the way a device would."""
        rng = np.random.default_rng(seed)
        total = int(round(seconds * self.samplerate))
        sent = 0
        while sent < total:
            n = min(self.blocksize, total - sent)
            block = (rng.standard_normal((n, self.channels)) * level).astype(np.float32)
            self.callback(block, n, None, None)
            sent += n
        self.delivered += sent
        return sent

    @property
    def released(self) -> bool:
        return self.stopped and self.closed


class FakeInput:
    """A factory the recorder opens streams from; keeps every stream made."""

    def __init__(self, fail_with: Exception = None):
        self.streams = []
        self.fail_with = fail_with

    def __call__(self, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream

    @property
    def last(self) -> FakeStream:
        return self.streams[-1]


def _fake_input(app, fail_with=None) -> FakeInput:
    factory = FakeInput(fail_with)
    app.recorder = TakeRecorder(open_stream=factory, sample_rate=app.sample_rate)
    return factory


def _a_tune(app, title):
    app.new_project(title, write=False)
    app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                     situation="A hopeful reunion after a long separation",
                     notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                           "Charanam and Ending.")
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=37)
    deadline = time.time() + 60
    while (app.project.melody() is None or app.jobs.active_jobs()) and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    assert app.project.melody() is not None, app.status_text


def _pallavi(app):
    return next(s for s in app.project.melody().sections if s.kind is SectionKind.PALLAVI)


# ----------------------------------------------------------------------
def test_record_then_stop_keeps_the_take_where_it_was_made(app):
    _a_tune(app, "A take of my own")
    mic = _fake_input(app)
    pallavi = _pallavi(app)
    renders_before = list(app.project.vocal_renders)
    melodies_before = len(app.project.melodies)

    assert app.start_take(section_id=pallavi.id)
    assert app.recorder.recording and mic.last.started
    assert "Recording Pallavi" in app.recording_status()
    assert "Recording Pallavi" in app.status_text
    delivered = mic.last.feed(1.5)
    assert app.recorder.recording, "Stop, not the audio, ends a take"

    take = app.stop_take()
    assert take is not None, app.status_text
    assert not app.recorder.recording and mic.last.released
    assert take.duration == pytest.approx(delivered / app.sample_rate)
    assert take.sample_rate == app.sample_rate
    assert take.section_id == pallavi.id and take.section_name == "Pallavi"
    assert take.start == pytest.approx(pallavi.start) and take.end == pytest.approx(pallavi.end)
    assert take.melody_version == app.project.melody().version
    assert take.label == "Take 1 - Pallavi"
    assert app.project.recordings == [take]

    audio, sr = sf.read(take.audio_path, dtype="float32", always_2d=False)
    assert sr == app.sample_rate and len(audio) == delivered
    assert "audio" in take.audio_path.replace("\\", "/").split("/")
    assert float(np.abs(audio).max()) > 0.01, "the file holds the audio the stream gave"

    # Nothing rendered or composed moved.
    assert app.project.vocal_renders == renders_before
    assert len(app.project.melodies) == melodies_before
    assert app.project.vocal_master_id == ""
    assert "Kept Take 1 - Pallavi" in app.status_text


def test_a_second_take_preserves_the_first(app):
    _a_tune(app, "Two takes")
    mic = _fake_input(app)
    app.start_take()
    mic.last.feed(0.5)
    first = app.stop_take()
    app.start_take()
    mic.last.feed(0.8)
    second = app.stop_take()
    assert first is not None and second is not None
    assert [t.id for t in app.project.recordings] == [first.id, second.id]
    assert first.audio_path != second.audio_path
    assert second.label == "Take 2 - whole song"
    kept, _ = sf.read(first.audio_path, dtype="float32")
    assert len(kept) == int(0.5 * app.sample_rate)
    assert all(s.released for s in mic.streams) and len(mic.streams) == 2


def test_cancel_keeps_nothing_and_releases_the_input(app):
    _a_tune(app, "Cancelled take")
    mic = _fake_input(app)
    assert app.start_take()
    mic.last.feed(1.0)
    app.cancel_take()
    assert not app.recorder.recording and mic.last.released
    assert app.project.recordings == []
    assert "cancelled" in app.status_text.lower() and "nothing" in app.status_text.lower()
    # A Stop after a Cancel is not a take either.
    assert app.stop_take() is None
    assert app.project.recordings == []


def test_a_device_that_will_not_open_is_an_error_state_not_a_crash(app):
    _a_tune(app, "Bad device")
    mic = _fake_input(app, fail_with=OSError("Error opening InputStream: device unavailable"))
    assert not app.start_take()
    assert not app.recorder.recording
    assert app.recorder.state.phase == "error"
    assert "device unavailable" in app.recorder.state.error
    assert "input device" in app.recorder.state.error.lower()
    assert "not in use by another application" in app.recorder.state.error
    assert app.project.recordings == [] and not mic.streams
    assert "Input error" in app.recording_status()


def test_without_a_backend_record_is_refused_with_what_to_do(app):
    _a_tune(app, "No backend")
    app.recorder = TakeRecorder(backend_present=False, sample_rate=app.sample_rate)
    assert not app.recorder.available
    assert not app.start_take()
    assert app.recording_status().startswith("Input unavailable")
    assert "sounddevice" in app.recorder.state.error and "install" in app.recorder.state.error.lower()
    assert app.project.recordings == []
    assert app.stop_take() is None


def test_switching_songs_releases_the_input_and_keeps_nothing(app):
    _a_tune(app, "First song")
    mic = _fake_input(app)
    assert app.start_take()
    mic.last.feed(0.7)
    app.new_project("Second song", write=False)
    assert not app.recorder.recording and mic.last.released
    assert app.project.recordings == []
    # The old stream's late blocks are refused, and Stop is not a take.
    mic.last.feed(0.3)
    assert app.stop_take() is None
    assert "Not recording" in app.status_text
    assert app.project.recordings == []


def test_a_take_stopped_in_another_song_is_not_kept_there(app, tmp_path):
    """A stop that arrives with a ticket from another song - the song
    changed under it - keeps nothing, whichever song is open."""
    _a_tune(app, "Ticketed song")
    mic = _fake_input(app)
    assert app.start_take()
    mic.last.feed(0.4)
    # Simulate the song changing under a still-open stream without going
    # through new_project: the ticket names a project that is gone.
    app._take_ticket["project"] = "proj_somewhere_else"
    assert app.stop_take() is None
    assert "song changed" in app.status_text
    assert app.project.recordings == [] and mic.last.released


def test_closing_the_application_releases_the_input(settings):
    from raagacomposer.app import AppController
    controller = AppController(settings)
    try:
        controller.new_project("Closing", write=False)
        mic = _fake_input(controller)
        assert controller.start_take()
        mic.last.feed(0.2)
    finally:
        controller.close()
    assert not controller.recorder.recording and mic.last.released


def test_an_empty_take_is_not_kept(app):
    _a_tune(app, "Silence")
    mic = _fake_input(app)
    assert app.start_take()
    assert app.stop_take() is None
    assert "Nothing was recorded" in app.status_text
    assert app.project.recordings == [] and mic.last.released


def test_play_take_plays_only_that_file(app, monkeypatch):
    _a_tune(app, "Play it back")
    mic = _fake_input(app)
    app.start_take()
    mic.last.feed(0.6)
    take = app.stop_take()
    loaded = {}

    def load(audio, sample_rate=None, kind=None):
        loaded.update(samples=len(audio), rate=sample_rate, kind=kind)
    monkeypatch.setattr(app.playback, "load", load)
    monkeypatch.setattr(app.playback, "play", lambda *a, **k: True)
    assert app.play_take(take.id)
    assert loaded == {"samples": int(0.6 * app.sample_rate), "rate": app.sample_rate,
                      "kind": "take"}
    assert "recorded by you" in app.status_text
    assert not app.play_take("rec_nothing")
    assert app.project.vocal_renders == [] and len(app.project.recordings) == 1


def test_a_kept_take_survives_saving_and_reopening_the_song(app):
    _a_tune(app, "Saved take")
    mic = _fake_input(app)
    app.start_take(section_id=_pallavi(app).id)
    mic.last.feed(0.3)
    take = app.stop_take()
    app.save()
    reopened = app.store.open(app.project_dir)
    assert [t.id for t in reopened.recordings] == [take.id]
    kept = reopened.recordings[0]
    assert (kept.section_name, kept.audio_path, kept.duration, kept.melody_version) == \
        ("Pallavi", take.audio_path, take.duration, take.melody_version)


def test_the_recorder_alone_holds_the_input_only_between_start_and_stop():
    factory = FakeInput()
    recorder = TakeRecorder(open_stream=factory, sample_rate=8000, block=100)
    assert recorder.available and not recorder.recording
    assert recorder.stop() is None and recorder.status_text() == "Not recording"
    assert recorder.start(device="Fake Mic")
    assert recorder.session == 1 and factory.last.device == "Fake Mic"
    assert factory.last.samplerate == 8000 and factory.last.blocksize == 100
    factory.last.feed(0.25)
    audio = recorder.stop()
    assert len(audio) == 2000 and audio.dtype == np.float32
    assert factory.last.released
    # A block arriving after the stop is dropped, not kept for the next take.
    factory.last.feed(0.1)
    assert recorder.start()
    assert recorder.session == 2
    recorder.cancel()
    assert recorder.stop() is None and factory.last.released
    recorder.close()


def test_without_sounddevice_the_recorder_says_so():
    recorder = TakeRecorder(backend_present=False)
    assert not recorder.available
    assert not recorder.start()
    assert recorder.state.phase == "off" and recorder.state.error == NO_BACKEND
    assert recorder.status_text().startswith("Input unavailable: ")
