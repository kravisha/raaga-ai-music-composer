"""The Agentic Band's stage (studio/band.py, ui/panels/band_panel.py)."""
from __future__ import annotations

import os
import time

import pytest

from raagacomposer.studio import band as B

pytestmark = pytest.mark.unit


def test_every_seat_shows_an_instrument_or_a_coloured_mic():
    seats = B.default_band()
    assert any(s.agent == "Kali" and s.role == B.VOCAL and s.title == "lead producer" for s in seats)
    for s in seats:
        assert s.icon
        if s.role == B.VOCAL:
            assert s.mic_colour in (B.MIC_COLOURS["male"], B.MIC_COLOURS["female"])
        else:
            assert s.instrument in B.PLAYABLE
    assert B.MIC_COLOURS["male"] != B.MIC_COLOURS["female"]
    assert B.Seat("x", B.VOCAL, voice="male").mic_colour == B.MIC_COLOURS["male"]   # blue
    assert B.Seat("y", B.VOCAL, voice="female").mic_colour == B.MIC_COLOURS["female"]   # pink


def test_only_the_seats_that_are_playing_are_heard():
    seats = B.default_band()
    heard, lead, _ = B.arrangement(seats)
    assert "saxophone" in heard and lead, "Bhairava's sax and Kali's voice are on"
    for s in seats:
        if s.agent in ("Bhairava", "Kali"):
            s.playing = False
    heard, lead, _ = B.arrangement(seats)
    assert "saxophone" not in heard
    assert lead == []


def test_an_agent_changing_instrument_changes_what_is_heard():
    seats = [B.Seat("Loki", B.INSTRUMENT, "violin")]
    heard, _, _ = B.arrangement(seats)
    assert set(heard) == {"violin"}
    seats[0].instrument = "veena"
    heard, _, _ = B.arrangement(seats)
    assert set(heard) == {"veena"}


def test_the_band_renders_as_stereo_audio():
    seats = [B.Seat("Jingjing", B.INSTRUMENT, "drum_kit"), B.Seat("Kali", B.VOCAL, voice="female")]
    audio = B.render(seats, sr=16000)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert float(abs(audio).max()) > 0.01


@pytest.mark.ui
def test_the_stage_counts_down_before_it_plays(app, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from raagacomposer.ui.panels import band_panel as BP

    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(BP, "COUNTDOWN_S", 2)
    played = []
    monkeypatch.setattr(app, "play_render", lambda kind=None, **k: played.append(kind) or True)
    panel = BP.BandPanel(app)
    assert len(panel.cards) == len(panel.seats) >= 8
    for card in panel.cards[2:]:
        card.playing.setChecked(False)
    panel.count_in()
    assert "Get ready" in panel.status.text()
    assert played == [], "nothing plays before the countdown ends"
    deadline = time.time() + 60
    while time.time() < deadline and not played:
        app.pump()
        qt.processEvents()
        time.sleep(0.05)
    assert played == ["band"]
    assert "Playing" in panel.status.text()
    panel.close()
