"""Export engine (spec section 19).

WAV always, MP3 when ffmpeg is on the machine, plus symbolic exports (MIDI and
MusicXML) written directly so no extra dependency is required, plus a zipped
project archive.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import soundfile as sf

from ..core.logging_setup import get_logger
from ..core.models import ArrangementVersion, LyricsVersion, MelodyVersion, Note
from ..music.theory import midi_name

log = get_logger("export")


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def write_wav(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_24"
              ) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype=subtype)
    return path


def ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def write_mp3(path: Path, audio: np.ndarray, sr: int, bitrate: str = "256k"
              ) -> Path:
    """MP3 via ffmpeg. Raises with a clear message when ffmpeg is absent."""
    exe = ffmpeg_path()
    if not exe:
        raise RuntimeError(
            "MP3 export needs ffmpeg on PATH. The WAV export always works.")
    path = Path(path)
    tmp_wav = path.with_suffix(".tmp.wav")
    write_wav(tmp_wav, audio, sr, subtype="PCM_16")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(tmp_wav),
           "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        tmp_wav.unlink(missing_ok=True)
    return path


def export_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    suffix = Path(path).suffix.lower()
    if suffix == ".mp3":
        return write_mp3(path, audio, sr)
    return write_wav(path, audio, sr)


def export_stems(directory: Path, stems: Dict[str, np.ndarray], sr: int
                 ) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for name, audio in stems.items():
        safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in name)
        out.append(write_wav(directory / f"{safe}.wav", audio, sr))
    return out


# --------------------------------------------------------------------------
# MIDI
# --------------------------------------------------------------------------
def _vlq(value: int) -> bytes:
    value = max(0, int(value))
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def _midi_track(events: List[tuple], name: str = "") -> bytes:
    """events: (tick, status, data1, data2) sorted by tick."""
    data = bytearray()
    if name:
        payload = name.encode("utf-8")[:127]
        data += _vlq(0) + b"\xFF\x03" + bytes([len(payload)]) + payload
    last = 0
    for tick, status, d1, d2 in events:
        data += _vlq(int(tick - last))
        data += bytes([status, d1 & 0x7F, d2 & 0x7F])
        last = tick
    data += _vlq(0) + b"\xFF\x2F\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


def write_midi(path: Path, melody: MelodyVersion,
               arrangement: Optional[ArrangementVersion] = None,
               ticks_per_beat: int = 480) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpm = max(20, melody.tempo_bpm)
    sec_to_tick = ticks_per_beat * bpm / 60.0

    tracks: List[bytes] = []
    # Tempo map.
    tempo = int(60_000_000 / bpm)
    meta = bytearray()
    meta += _vlq(0) + b"\xFF\x51\x03" + tempo.to_bytes(3, "big")
    meta += _vlq(0) + b"\xFF\x2F\x00"
    tracks.append(b"MTrk" + struct.pack(">I", len(meta)) + bytes(meta))

    def note_events(notes: Iterable[Note], channel: int) -> List[tuple]:
        evs: List[tuple] = []
        for n in notes:
            on = int(n.start * sec_to_tick)
            off = max(on + 1, int(n.end * sec_to_tick))
            evs.append((on, 0x90 | channel, int(n.midi), int(n.velocity)))
            evs.append((off, 0x80 | channel, int(n.midi), 0))
        return sorted(evs, key=lambda e: (e[0], e[1] & 0xF0))

    tracks.append(_midi_track(note_events(melody.notes, 0), f"Tune {melody.raaga}"))

    if arrangement:
        for i, track in enumerate(arrangement.tracks[:14], start=1):
            channel = 9 if track.role == "rhythm" else min(15, i)
            notes = [n for r in track.regions for n in r.notes]
            if notes:
                tracks.append(_midi_track(note_events(notes, channel), track.label))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ticks_per_beat)
    path.write_bytes(header + b"".join(tracks))
    return path


# --------------------------------------------------------------------------
# MusicXML
# --------------------------------------------------------------------------
_STEP_ALTER = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0),
               5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0),
               10: ("B", -1), 11: ("B", 0)}


def write_musicxml(path: Path, melody: MelodyVersion,
                   lyrics: Optional[LyricsVersion] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    divisions = 4
    beat = 60.0 / max(20, melody.tempo_bpm)
    syllable_for: Dict[int, str] = {}
    if lyrics:
        for line in lyrics.lines:
            for idx, syl in zip(line.note_indices, line.syllables):
                syllable_for[idx] = syl.lstrip("~")

    rows: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 '
        'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        "  <work><work-title>Raaga Composer tune - "
        f"{melody.raaga}</work-title></work>",
        "  <part-list><score-part id=\"P1\">"
        "<part-name>Voice</part-name></score-part></part-list>",
        '  <part id="P1">',
    ]
    measure_len = melody.beats_per_cycle * beat
    measures: Dict[int, List[Note]] = {}
    for i, n in enumerate(melody.notes):
        measures.setdefault(int(n.start // max(0.1, measure_len)) + 1, []).append((i, n))

    for number in sorted(measures):
        rows.append(f'    <measure number="{number}">')
        if number == 1:
            rows.append("      <attributes>"
                        f"<divisions>{divisions}</divisions>"
                        "<key><fifths>0</fifths></key>"
                        f"<time><beats>{melody.beats_per_cycle}</beats>"
                        "<beat-type>4</beat-type></time>"
                        "<clef><sign>G</sign><line>2</line></clef>"
                        "</attributes>")
        for idx, n in measures[number]:
            step, alter = _STEP_ALTER[int(n.midi) % 12]
            octave = int(n.midi) // 12 - 1
            dur = max(1, int(round(n.duration / beat * divisions)))
            rows.append("      <note>")
            rows.append(f"        <pitch><step>{step}</step>"
                        + (f"<alter>{alter}</alter>" if alter else "")
                        + f"<octave>{octave}</octave></pitch>")
            rows.append(f"        <duration>{dur}</duration>")
            rows.append(f"        <type>{_note_type(dur, divisions)}</type>")
            syl = syllable_for.get(idx)
            if syl:
                rows.append("        <lyric><syllabic>single</syllabic>"
                            f"<text>{_xml_escape(syl)}</text></lyric>")
            rows.append("      </note>")
        rows.append("    </measure>")
    rows.append("  </part>")
    rows.append("</score-partwise>")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _note_type(duration: int, divisions: int) -> str:
    ratio = duration / divisions
    for limit, name in ((0.2, "32nd"), (0.4, "16th"), (0.8, "eighth"),
                        (1.5, "quarter"), (3.0, "half")):
        if ratio <= limit:
            return name
    return "whole"


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------
# text and archive
# --------------------------------------------------------------------------
def write_lyrics_text(path: Path, lyrics: LyricsVersion,
                      melody: Optional[MelodyVersion] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    current = ""
    for line in lyrics.lines:
        section = melody.section_by_id(line.section_id) if melody else None
        name = section.name if section else ""
        if name and name != current:
            rows.append(f"\n[{name}]")
            current = name
        rows.append(f"{line.text}")
    path.write_text("\n".join(rows).strip() + "\n", encoding="utf-8")
    return path


def archive_project(path: Path, project_dir: Path) -> Path:
    """Zip the whole project folder: decisions, references and artifacts."""
    path = Path(path)
    project_dir = Path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in project_dir.rglob("*"):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f.relative_to(project_dir))
    return path
