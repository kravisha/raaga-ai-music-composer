"""The Jam Room (Krish, 7 Nov 2026).

"A separate tab called jam room. That's where we jam. Agents represented by instruments ... they should be able to connect ... listen
to what is asked real time. One orchestrator - for now it will be me - will say 1, 2, 3, go, and then your music will start."
"We need a lyricist, one lead singer, one for the beat, at least a couple of people for instruments ... and one for a tambura. If you
can make it auto, that's fine.  The minimum is 3: singer, lead, one instrument."  Rhythm can be any of drums, tabla, mridangam,
ghatam, bongos ("I like the bongos for certain kind of music").

A seat is data: who sits there, which ROLE they have and what they play.  The music is the groove from ``groove.orchestrate_show``,
handed out by role:  the lead plays the hook, a second instrument the counter-line (stabs, the run, the line an octave up), the bass
the bass line, each rhythm seat its own pattern, the drone the tanpura (automatic when nobody sits there), the singer sings the
hook on "aa".  The lyricist's lines are written from the mood and shown to the room.  The conductor counts the band in: 1, 2, 3, go.

Agents join from anywhere by posting to the team board (``JAM JOIN <role> [instrument] [male|female]``); the room posts the
count-in, so every agent on the board hears it at the same moment.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..audio import dsp
from ..core.models import MelodyVersion, Note, Section, SectionKind, VocalDirection
from ..music import instruments as catalog
from ..music.synth import render_notes
from . import groove as G

CONDUCTOR, LYRICIST, SINGER, LEAD, SECOND, BASS, RHYTHM, DRONE = (
    "conductor", "lyricist", "singer", "lead", "second", "bass", "rhythm", "drone")
ROLES = (CONDUCTOR, LYRICIST, SINGER, LEAD, SECOND, BASS, RHYTHM, DRONE)
ROLE_NAMES = {CONDUCTOR: "Conductor (1, 2, 3, go)", LYRICIST: "Lyricist", SINGER: "Lead singer", LEAD: "Lead instrument",
              SECOND: "Second instrument", BASS: "Bass", RHYTHM: "Rhythm / beat", DRONE: "Drone (tanpura)"}

MELODIC = ("saxophone", "violin", "veena", "flute", "shehnai", "sitar", "santoor", "sarod", "clarinet", "guitar",
           "electric_guitar", "piano", "electric_piano", "harmonium", "strings", "cello", "synth_pad")
PERCUSSION = ("drum_kit", "mridangam", "tabla", "ghatam", "bongos", "kanjira", "tambourine")
#: what each role may play; a role with no list plays nothing (conductor, lyricist) or sings (singer)
CHOICES = {LEAD: MELODIC, SECOND: MELODIC, BASS: ("bass", "double_bass"), RHYTHM: PERCUSSION, DRONE: ("tanpura",)}

ICONS = {CONDUCTOR: "\U0001FA84", LYRICIST: "✍️", SINGER: "\U0001F3A4", DRONE: "\U0001FA95",
         "saxophone": "\U0001F3B7", "violin": "\U0001F3BB", "cello": "\U0001F3BB", "strings": "\U0001F3BB", "veena": "\U0001FA95",
         "sitar": "\U0001FA95", "sarod": "\U0001FA95", "santoor": "\U0001FA95", "flute": "\U0001FA88", "shehnai": "\U0001FA88",
         "clarinet": "\U0001FA88", "guitar": "\U0001F3B8", "electric_guitar": "\U0001F3B8", "bass": "\U0001F3B8",
         "double_bass": "\U0001F3BB", "piano": "\U0001F3B9", "electric_piano": "\U0001F3B9", "harmonium": "\U0001F3B9",
         "synth_pad": "\U0001F3B9", "drum_kit": "\U0001F941", "mridangam": "\U0001FA98", "tabla": "\U0001FA98",
         "ghatam": "\U0001FAD9", "bongos": "\U0001FA98", "kanjira": "\U0001FA98", "tambourine": "\U0001FA98"}
MIC_COLOURS = {"male": "#3b82f6", "female": "#ec4899", "neutral": "#a3a3a3"}

#: rhythm patterns over one bar of 16 steps: (step, stroke, velocity); the stroke picks the drum's hit (0 = its lowest)
PATTERNS = {
    "mridangam": ((0, 0, 110), (3, 2, 70), (4, 1, 92), (6, 2, 60), (8, 0, 100), (10, 3, 72), (11, 2, 60), (12, 1, 95),
                  (14, 3, 72), (15, 2, 55)),
    "tabla": ((0, 0, 100), (2, 2, 60), (4, 3, 86), (6, 2, 60), (7, 1, 70), (8, 0, 96), (10, 2, 60), (12, 3, 86),
              (14, 1, 70), (15, 2, 55)),
    "ghatam": ((0, 0, 92), (2, 1, 60), (5, 2, 72), (8, 0, 86), (10, 1, 60), (11, 2, 55), (13, 1, 72), (14, 2, 60)),
    "bongos": ((0, 1, 96), (2, 1, 56), (4, 2, 86), (6, 1, 56), (8, 1, 92), (10, 1, 56), (12, 0, 86), (14, 0, 76), (15, 1, 50)),
    "kanjira": ((0, 0, 92), (3, 1, 60), (6, 2, 72), (8, 0, 86), (11, 1, 60), (12, 2, 76), (14, 1, 56)),
    "tambourine": ((2, 0, 66), (6, 0, 66), (10, 0, 66), (14, 0, 66), (5, 1, 36), (13, 1, 36)),
}
MIX = {LEAD: (0.95, 0.08), SECOND: (0.6, -0.2), BASS: (0.9, 0.0), DRONE: (0.3, -0.1), SINGER: (0.85, 0.0)}
RHYTHM_MIX = {"drum_kit": (1.0, 0.0), "mridangam": (0.65, -0.15), "tabla": (0.6, 0.2), "ghatam": (0.55, -0.3),
              "bongos": (0.6, 0.3), "kanjira": (0.4, -0.3), "tambourine": (0.32, 0.32)}

COUNT = ("1", "2", "3", "go")


@dataclass
class JamSeat:
    agent: str
    role: str = LEAD
    instrument: str = ""
    voice: str = "neutral"          # a singer's mic: male (blue) / female (pink)
    playing: bool = True
    joined: str = "room"            # "room" = seated here; "board" = joined from the team board
    seen: float = 0.0               # last time the agent was heard on the board (epoch seconds)

    @property
    def icon(self) -> str:
        return ICONS.get(self.role if self.role in (CONDUCTOR, LYRICIST, SINGER) else self.instrument, "\U0001F3B5")

    @property
    def mic_colour(self) -> str:
        return MIC_COLOURS.get(self.voice, MIC_COLOURS["neutral"])

    def to_dict(self) -> dict:
        return asdict(self)


def default_room() -> list[JamSeat]:
    """The seating the room talked about on 7 Nov: Krish counts in, Kali sings, Arya on veena, Alfred on light percussion.
    Agents change their own seats (or join from the board); these are suggestions."""
    return [
        JamSeat("Krish", CONDUCTOR, voice="male"),
        JamSeat("Kali", SINGER, voice="female"),
        JamSeat("Claude Work", LYRICIST),
        JamSeat("Bhairava", LEAD, "saxophone"),
        JamSeat("Loki", SECOND, "violin"),
        JamSeat("Arya", SECOND, "veena", playing=False),
        JamSeat("Tara", BASS, "bass"),
        JamSeat("Jingjing", RHYTHM, "drum_kit"),
        JamSeat("Boris", RHYTHM, "mridangam"),
        JamSeat("Diego", RHYTHM, "bongos"),
        JamSeat("Alfred", RHYTHM, "tambourine", playing=False),
    ]


def fix_seat(seat: JamSeat) -> JamSeat:
    """Keep a seat consistent: an instrument its role can play, no instrument for a conductor, lyricist or singer."""
    allowed = CHOICES.get(seat.role)
    if allowed:
        if seat.instrument not in allowed:
            seat.instrument = allowed[0]
    else:
        seat.instrument = ""
    return seat


def check(seats: list[JamSeat]) -> tuple[bool, list[str]]:
    """Krish's minimum: a singer, a lead instrument and one more instrument.  -> (ready, notes for the room)."""
    on = [s for s in seats if s.playing]
    has = {r: [s for s in on if s.role == r] for r in ROLES}
    notes: list[str] = []
    others = has[SECOND] + has[BASS] + has[RHYTHM]
    if not has[SINGER]:
        notes.append("Needs a lead singer.")
    if not has[LEAD]:
        notes.append("Needs a lead instrument.")
    if not others:
        notes.append("Needs one more instrument (second, bass or rhythm).")
    ready = not notes
    if len(has[CONDUCTOR]) > 1:
        notes.append("Two conductors: only " + has[CONDUCTOR][0].agent + " counts in.")
    if not has[CONDUCTOR]:
        notes.append("No conductor: the room counts itself in.")
    if not has[RHYTHM]:
        notes.append("No beat yet: add a rhythm seat (drums, tabla, mridangam, ghatam, bongos).")
    if not has[LYRICIST]:
        notes.append("No lyricist: the singer sings 'aa'.")
    if not has[DRONE]:
        notes.append("Tanpura on auto.")
    return ready, notes


def _bar_times() -> tuple[float, float, float, float]:
    """(seconds per step, seconds per bar, groove start, groove end) of the show with no silence before it."""
    g = G.build()
    st = g.step_seconds
    bar_s = G.STEPS * st
    start = G.ENTRANCE_BARS * bar_s
    return st, bar_s, start, start + (g.bars - g.outro_bars) * bar_s


def _shift(notes: list[Note], by: float, semis: int = 0) -> list[Note]:
    return [Note(swara=n.swara, midi=n.midi + semis, start=round(n.start + by, 4), duration=n.duration, velocity=n.velocity)
            for n in notes]


def _stroke_midi(inst_key: str, stroke: int) -> int:
    inst = catalog.get(inst_key)
    n = max(1, len(inst.hit_freqs)) if inst else 1
    return 48 + (stroke % n)          # 48 is a multiple of 2, 3 and 4: the synth picks hit (midi % n)


def rhythm_part(inst_key: str, parts: dict[str, list[Note]]) -> list[Note]:
    """What a rhythm seat plays under the groove."""
    if inst_key == "drum_kit":
        return list(parts.get("drum_kit", []))
    st, bar_s, start, end = _bar_times()
    pattern = PATTERNS.get(inst_key, PATTERNS["tabla"])
    out: list[Note] = []
    t, bar = start, 0
    while t < end - 1e-6:
        for step, stroke, vel in pattern:
            out.append(Note(swara="-", midi=_stroke_midi(inst_key, stroke), start=round(t + step * st, 4), duration=st,
                            velocity=min(127, vel + (8 if bar % 4 == 0 and step == 0 else 0))))
        if inst_key == "mridangam" and bar % 4 == 3:          # the fill into every fifth bar
            out.extend(Note(swara="-", midi=_stroke_midi(inst_key, i), start=round(t + s * st, 4), duration=st, velocity=70 + 7 * i)
                       for i, s in enumerate((12, 13, 14, 15)))
        t += bar_s
        bar += 1
    return out


def count_in_seconds() -> float:
    return 4 * 60.0 / G.BPM          # four beats: 1, 2, 3, go


def count_in_notes() -> list[Note]:
    """The conductor's 1, 2, 3 (a stick click each) and GO (kick and clap together)."""
    beat = 60.0 / G.BPM
    out = [Note(swara="-", midi=50, start=round(i * beat, 4), duration=0.08, velocity=96) for i in range(3)]
    out += [Note(swara="-", midi=48, start=round(3 * beat, 4), duration=0.2, velocity=120),
            Note(swara="-", midi=49, start=round(3 * beat, 4), duration=0.2, velocity=110)]
    return out


def tanpura(seconds: float) -> list[Note]:
    from ..raaga.library import library
    from .rough_cut import tanpura_notes
    raaga = library().require(G.Groove().raaga)
    return tanpura_notes(G.TONIC - 12, seconds, list(raaga.arohanam) + list(raaga.avarohanam))


def arrangement(seats: list[JamSeat]) -> dict:
    """Who plays which notes. -> {"parts": [(seat, instrument, notes, gain, pan)], "singers": [...], "lead": notes,
    "count_in": seconds or 0, "conductor": name or "", "seconds": total, "auto_drone": bool}."""
    parts, seconds = G.orchestrate_show(0.0)
    on = [s for s in seats if s.playing]
    conductors = [s for s in on if s.role == CONDUCTOR]
    lead_in = count_in_seconds()                      # the room counts itself in when nobody conducts
    counter = sorted(parts.get("strings", []) + parts.get("violin", []) + parts.get("veena", []), key=lambda n: n.start)
    out: list[tuple[JamSeat | None, str, list[Note], float, float]] = []
    out.append((conductors[0] if conductors else None, "drum_kit", count_in_notes(), 0.8, 0.0))
    for s in on:
        if s.role == LEAD:
            gain, pan = MIX[LEAD]
            out.append((s, s.instrument, _shift(parts.get("saxophone", []), lead_in), gain, pan))
        elif s.role == SECOND:
            gain, pan = MIX[SECOND]
            out.append((s, s.instrument, _shift(counter, lead_in), gain, pan))
        elif s.role == BASS:
            gain, pan = MIX[BASS]
            out.append((s, s.instrument, _shift(parts.get("bass", []), lead_in), gain, pan))
        elif s.role == RHYTHM:
            gain, pan = RHYTHM_MIX.get(s.instrument, (0.5, 0.0))
            out.append((s, s.instrument, _shift(rhythm_part(s.instrument, parts), lead_in), gain, pan))
    drones = [s for s in on if s.role == DRONE]
    gain, pan = MIX[DRONE]
    out.append((drones[0] if drones else None, "tanpura", _shift(tanpura(seconds), lead_in), gain, pan))
    return {"parts": out, "singers": [s for s in on if s.role == SINGER], "lead": _shift(parts.get("saxophone", []), lead_in),
            "count_in": lead_in, "conductor": conductors[0].agent if conductors else "", "seconds": lead_in + seconds,
            "auto_drone": not drones}


def render(seats: list[JamSeat], sr: int = 44100, seed: int = 11) -> np.ndarray:
    """The jam as stereo audio: the count-in, then every playing seat."""
    from ..providers.local import LocalVoiceProvider
    from ..voice.profiles import BUILTIN

    plan = arrangement(seats)
    total = plan["seconds"] + 1.5
    mix = dsp.silence(total, sr)
    for k, (_seat, key, notes, gain, pan) in enumerate(plan["parts"]):
        if not notes:
            continue
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        audio = render_notes(notes, inst, sr, total_seconds=total, seed=seed + k)
        mix = dsp.mix_into(mix, dsp.pan_mono(audio * gain, pan), 0)
    singers = plan["singers"]
    if singers and plan["lead"]:
        voices = {p.gender: p for p in BUILTIN}
        for k, singer in enumerate(singers):
            profile = voices.get(singer.voice) or BUILTIN[0]
            semis = -12 if profile.gender == "male" else 0
            melody = MelodyVersion(raaga=G.Groove().raaga, tonic_midi=G.TONIC + semis, tempo_bpm=G.BPM,
                                   notes=_shift(plan["lead"], 0.0, semis),
                                   sections=[Section(id="jam", kind=SectionKind.PALLAVI, start=0.0, end=total)])
            sung = LocalVoiceProvider().render_vocal(melody, None, profile, VocalDirection(style="energetic"), sr, total,
                                                     seed=seed + 50 + k, vocal_sections_only=False, vowel="a")
            span = 0.0 if len(singers) == 1 else (-0.2 + 0.4 * k / (len(singers) - 1))
            gain, _ = MIX[SINGER]
            mix = dsp.mix_into(mix, dsp.pan_mono(sung * gain, span), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.0, sr).astype(np.float32)


def lyricist_lines(mood: str, language: str = "English") -> list[str]:
    """The lyricist's seat writes two lines from the mood (the song team's own lyricist, no model needed)."""
    from .team import Lyricist, MoodListener

    card = MoodListener().listen(mood or "a joyful jam with friends", language)
    return list(Lyricist().write(card).lines)


# --------------------------------------------------------------------------
# The team board: how agents anywhere connect to the room and hear the count-in
# --------------------------------------------------------------------------
BOARD = "http://127.0.0.1:8799"
JAM_RE = re.compile(r"^\s*JAM\s+(JOIN|LEAVE|HERE)\b\s*(.*)$", re.IGNORECASE)
ROLE_WORDS = {"conductor": CONDUCTOR, "orchestrator": CONDUCTOR, "lyricist": LYRICIST, "lyrics": LYRICIST, "singer": SINGER,
              "vocal": SINGER, "vocals": SINGER, "lead": LEAD, "second": SECOND, "bass": BASS, "rhythm": RHYTHM, "beat": RHYTHM,
              "drums": RHYTHM, "drone": DRONE, "tanpura": DRONE, "tambura": DRONE}


def _at(msg: dict) -> float:
    try:
        return float(calendar.timegm(time.strptime(str(msg.get("at", ""))[:19], "%Y-%m-%dT%H:%M:%S")))   # the board's times are UTC
    except (ValueError, OverflowError):
        return 0.0


def parse_join(text: str) -> tuple[str, dict] | None:
    """'JAM JOIN rhythm bongos' -> ("join", {"role": "rhythm", "instrument": "bongos"}); 'JAM JOIN singer male';
    'JAM JOIN tabla' (an instrument alone picks its role); 'JAM LEAVE'; 'JAM HERE' (I am listening)."""
    m = JAM_RE.match(text or "")
    if not m:
        return None
    verb, rest = m.group(1).lower(), m.group(2).lower()
    if verb != "join":
        return verb, {}
    words = re.findall(r"[a-z_]+", rest.replace("drum kit", "drum_kit").replace("double bass", "double_bass"))
    info: dict = {}
    for w in words:
        if w in ("male", "female"):
            info["voice"] = w
        elif w in ROLE_WORDS and "role" not in info:
            info["role"] = ROLE_WORDS[w]
            if w == "bass":
                info.setdefault("instrument", "bass")
            if w == "drums":
                info.setdefault("instrument", "drum_kit")
            if w in ("tanpura", "tambura"):
                info.setdefault("instrument", "tanpura")
        elif catalog.get(w):
            info["instrument"] = w
    if "role" not in info:
        inst = info.get("instrument", "")
        info["role"] = (RHYTHM if inst in PERCUSSION else BASS if inst in CHOICES[BASS] else DRONE if inst == "tanpura"
                        else LEAD if inst else "")
    return ("join", info) if info.get("role") else None


def apply_board(seats: list[JamSeat], messages: list[dict], since: float, done: set[str] | None = None) -> list[str]:
    """Seat the agents who joined on the board since ``since``.  ``done`` holds the ids already applied, so a message acts once
    and the director's own changes on a card are never undone by an old post.  Returns what changed, for the status line.
    (The board does not check who posted: on this local board a name is trusted.)"""
    changes: list[str] = []
    for msg in messages:
        at = _at(msg)
        if at < since:
            continue
        key = str(msg.get("id") or f"{msg.get('at')}|{msg.get('from')}|{msg.get('text')}")
        if done is not None:
            if key in done:
                continue
            done.add(key)
        got = parse_join(str(msg.get("text", "")))
        who = str(msg.get("from", "")).strip()
        if not got or not who or who == "Jam Room":
            continue
        verb, info = got
        seat = next((s for s in seats if s.agent.lower() == who.lower()), None)
        if verb == "leave":
            if seat and seat.playing:
                seat.playing = False
                changes.append(f"{who} left")
            continue
        if seat is None:
            if verb != "join":                   # HERE from someone with no seat: nothing to show yet
                continue
            seat = JamSeat(who, info["role"], joined="board")
            seats.append(seat)
            changes.append(f"{who} joined")
        if verb == "join":
            seat.role = info["role"]
            seat.instrument = info.get("instrument", seat.instrument)
            seat.voice = info.get("voice", seat.voice)
            seat.playing = True
            fix_seat(seat)
        seat.seen = max(seat.seen, at)
    return changes


#: the board's own store (team_chat.py); reading its last ~256 KB is far cheaper than the whole board over HTTP (~10 MB)
BOARD_STORE = Path(os.environ.get("RAAGA_BOARD_STORE", r"C:\Users\krish\Strata-workspace\team-chat\messages.jsonl"))
TAIL_BYTES = 256 * 1024


def board_read(base: str = BOARD, timeout: float = 4.0, store: Path | None = None) -> list[dict]:
    """The board's latest messages: from the tail of its store file when this PC has it, else over HTTP."""
    store = store or BOARD_STORE
    if store.is_file():
        with store.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - TAIL_BYTES))
            chunk = fh.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()[1:] if size > TAIL_BYTES else chunk.splitlines()   # the first may be cut in half
        out: list[dict] = []
        for line in lines:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if isinstance(m, dict):
                out.append(m)
        return out
    with urllib.request.urlopen(base + "/api/messages", timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    msgs = data.get("messages", []) if isinstance(data, dict) else []
    return [m for m in msgs if isinstance(m, dict)][-400:]


def board_post(text: str, base: str = BOARD, timeout: float = 4.0) -> bool:
    req = urllib.request.Request(base + "/api/post", data=json.dumps({"from": "Jam Room", "to": ["All"], "text": text}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return bool(json.loads(r.read().decode("utf-8")).get("ok"))


def lineup(seats: list[JamSeat]) -> str:
    def one(s: JamSeat) -> str:
        what = s.instrument.replace("_", " ") if s.instrument else ROLE_NAMES[s.role].split(" (")[0].lower()
        return f"{s.agent} ({what})"
    return ", ".join(one(s) for s in seats if s.playing) or "nobody"
