"""The Jam Room (studio/jam.py, ui/panels/jam_panel.py)."""
from __future__ import annotations

import os
import time

import pytest

from raagacomposer.music import instruments as catalog
from raagacomposer.studio import jam as J

pytestmark = pytest.mark.unit


def _keys(plan):
    return {key for seat, key, notes, _g, _p in plan["parts"] if notes and seat is not None}


def test_the_default_room_has_the_roles_krish_asked_for():
    seats = J.default_room()
    roles = {s.role for s in seats if s.playing}
    assert {J.CONDUCTOR, J.LYRICIST, J.SINGER, J.LEAD, J.SECOND, J.RHYTHM} <= roles
    assert next(s for s in seats if s.role == J.CONDUCTOR).agent == "Krish"
    assert any(s.instrument == "bongos" for s in seats), "Krish: 'I like the bongos'"
    ok, notes = J.check(seats)
    assert ok, notes
    assert "Tanpura on auto." in notes


def test_the_minimum_is_a_singer_a_lead_and_one_more_instrument():
    assert not J.check([J.JamSeat("Kali", J.SINGER), J.JamSeat("B", J.LEAD, "saxophone")])[0]
    assert not J.check([J.JamSeat("B", J.LEAD, "saxophone"), J.JamSeat("J", J.RHYTHM, "tabla")])[0]
    ok, _ = J.check([J.JamSeat("Kali", J.SINGER), J.JamSeat("B", J.LEAD, "saxophone"), J.JamSeat("J", J.RHYTHM, "bongos")])
    assert ok


def test_a_seat_only_plays_what_its_role_allows():
    s = J.fix_seat(J.JamSeat("x", J.RHYTHM, "saxophone"))
    assert s.instrument in J.PERCUSSION
    s.role = J.LYRICIST
    assert J.fix_seat(s).instrument == ""
    for role, allowed in J.CHOICES.items():
        for key in allowed:
            assert catalog.get(key) is not None, f"{role} offers {key}, which the catalog cannot play"


def test_only_seats_in_the_jam_are_heard_and_the_drone_is_automatic():
    seats = [J.JamSeat("Kali", J.SINGER, voice="female"), J.JamSeat("B", J.LEAD, "flute"), J.JamSeat("D", J.RHYTHM, "bongos"),
             J.JamSeat("T", J.RHYTHM, "tabla", playing=False)]
    plan = J.arrangement(seats)
    assert _keys(plan) == {"flute", "bongos"}
    assert plan["auto_drone"] and any(k == "tanpura" and n for _s, k, n, _g, _p in plan["parts"])
    assert plan["singers"] and plan["lead"]
    assert plan["count_in"] == pytest.approx(4 * 60.0 / J.G.BPM)


def test_the_count_in_comes_first_and_the_band_after_it():
    plan = J.arrangement(J.default_room())
    count = next(n for s, k, n, _g, _p in plan["parts"] if k == "drum_kit" and s is not None and s.role == J.CONDUCTOR)
    assert len(count) == 5 and max(n.start for n in count) < plan["count_in"]
    for seat, key, notes, _g, _p in plan["parts"]:
        if seat is not None and seat.role != J.CONDUCTOR and key != "tanpura":
            assert min(n.start for n in notes) >= plan["count_in"] - 1e-6, key


def test_every_rhythm_pattern_hits_a_stroke_its_drum_has():
    for key in J.PERCUSSION:
        notes = J.rhythm_part(key, J.G.orchestrate_show(0.0)[0])
        assert notes, key
        inst = catalog.get(key)
        assert inst is not None, key
        for _step, stroke, _vel in J.PATTERNS.get(key, ()):
            assert stroke < len(inst.hit_freqs), f"{key} pattern asks for stroke {stroke}; the drum has {len(inst.hit_freqs)}"
        if key in J.PATTERNS:          # the written stroke is the one the synth plays (midi % hits)
            want = {stroke for _s, stroke, _v in J.PATTERNS[key]}
            assert {n.midi % len(inst.hit_freqs) for n in notes} >= want


def test_a_here_from_someone_without_a_seat_does_not_break_the_room():
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    seats = J.default_room()
    msgs = [{"id": "1", "from": "Nobody", "at": at, "text": "JAM HERE"}, {"id": "2", "from": "Ghost", "at": at, "text": "JAM LEAVE"},
            {"id": "3", "from": "Mimi", "at": at, "text": "JAM JOIN veena"}]
    J.apply_board(seats, msgs, since=time.time() - 600)
    names = {s.agent for s in seats}
    assert "Mimi" in names and "Nobody" not in names and "Ghost" not in names


def test_a_board_post_acts_once_so_the_director_can_change_it():
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    seats = J.default_room()
    done: set[str] = set()
    msgs = [{"id": "a", "from": "Boris", "at": at, "text": "JAM LEAVE"}]
    J.apply_board(seats, msgs, time.time() - 600, done)
    boris = next(s for s in seats if s.agent == "Boris")
    assert not boris.playing
    boris.playing = True                       # Krish seats him again
    J.apply_board(seats, msgs, time.time() - 600, done)   # the next poll sees the same post
    assert boris.playing


def test_the_board_is_read_from_the_tail_of_its_store(tmp_path):
    store = tmp_path / "messages.jsonl"
    rows = [{"id": str(i), "from": "x", "at": "2026-11-07T20:00:00Z", "text": "t" * 300} for i in range(2000)]
    store.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    got = J.board_read(store=store)
    assert got and got[-1]["id"] == "1999"
    assert len(got) < 2000, "only the tail is read"
    assert all(isinstance(m, dict) and "id" in m for m in got), "a half line at the cut is skipped"


def test_agents_join_from_the_board():
    now = time.time()
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 7200))
    seats = J.default_room()
    msgs = [{"from": "Mimi", "at": at, "text": "JAM JOIN second veena"},
            {"from": "Ganesha", "at": at, "text": "jam join tabla"},
            {"from": "Hanuman", "at": old, "text": "JAM JOIN rhythm ghatam"},       # before the room opened: ignored
            {"from": "Boris", "at": at, "text": "JAM LEAVE"},
            {"from": "Codex", "at": at, "text": "JAM JOIN singer male"},
            {"from": "Loki", "at": at, "text": "jamming is fun"}]                   # not a command
    changes = J.apply_board(seats, msgs, since=now - 600)
    by = {s.agent: s for s in seats}
    assert by["Mimi"].role == J.SECOND and by["Mimi"].instrument == "veena" and by["Mimi"].joined == "board"
    assert by["Ganesha"].role == J.RHYTHM and by["Ganesha"].instrument == "tabla"
    assert "Hanuman" not in by
    assert not by["Boris"].playing
    assert by["Codex"].role == J.SINGER and by["Codex"].voice == "male" and by["Codex"].mic_colour == J.MIC_COLOURS["male"]
    assert by["Loki"].role == J.SECOND
    assert "Mimi joined" in changes and "Boris left" in changes


def test_parse_join_understands_plain_words():
    def info(text):
        got = J.parse_join(text)
        assert got is not None, text
        return got[1]
    assert info("JAM JOIN drums") == {"role": J.RHYTHM, "instrument": "drum_kit"}
    assert info("JAM JOIN bass")["instrument"] == "bass"
    assert info("JAM JOIN tambura")["role"] == J.DRONE
    assert info("JAM JOIN lyricist")["role"] == J.LYRICIST
    assert J.parse_join("JAM HERE") == ("here", {})
    assert J.parse_join("hello JAM JOIN singer") is None


def test_the_jam_renders_as_stereo_audio():
    seats = [J.JamSeat("Kali", J.SINGER, voice="female"), J.JamSeat("B", J.LEAD, "saxophone"),
             J.JamSeat("D", J.RHYTHM, "bongos"), J.JamSeat("K", J.CONDUCTOR)]
    audio = J.render(seats, sr=16000)
    assert audio.ndim == 2 and audio.shape[1] == 2 and float(abs(audio).max()) > 0.01


def test_the_lyricist_writes_two_lines():
    lines = J.lyricist_lines("a joyful jam with friends")
    assert len(lines) == 2 and all(line.strip() for line in lines)


@pytest.mark.ui
def test_the_room_counts_down_then_1_2_3_go(app, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from raagacomposer.ui.panels import jam_panel as JP

    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(JP, "COUNTDOWN_S", 2)
    played = []
    monkeypatch.setattr(app, "play_render", lambda kind=None, **k: played.append(kind) or True)
    panel = JP.JamPanel(app, board=False)
    assert not panel.board_box.isChecked()
    keep = {"Kali", "Bhairava", "Diego", "Krish"}
    for card in panel.cards:
        card.playing.setChecked(card.seat.agent in keep)
    assert panel.go_btn.isEnabled()
    panel.count_in()
    assert "Get ready" in panel.big.text()
    assert not panel.go_btn.isEnabled(), "no second count-in while one is running"
    assert played == []
    deadline = time.time() + 90
    seen = set()
    while time.time() < deadline and (not played or "GO!" not in seen):
        app.pump()
        qt.processEvents()
        seen.add(panel.big.text())
        time.sleep(0.02)
    assert played == ["jam"]
    assert {"1", "2", "3", "GO!"} <= seen
    assert "Jamming" in panel.status.text()
    panel.stop()
    panel.close()


@pytest.mark.ui
def test_a_stopped_count_in_never_plays_late(app, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from raagacomposer.ui.panels import jam_panel as JP

    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(JP, "COUNTDOWN_S", 1)
    played = []
    monkeypatch.setattr(app, "play_render", lambda kind=None, **k: played.append(kind) or True)
    panel = JP.JamPanel(app, board=False)
    panel.count_in()
    panel.stop()
    deadline = time.time() + 30
    while time.time() < deadline and app.jobs.active_jobs():
        app.pump()
        qt.processEvents()
        time.sleep(0.05)
    for _ in range(40):
        app.pump()
        qt.processEvents()
        time.sleep(0.05)
    assert played == []
    assert panel.go_btn.isEnabled()
    panel.close()
