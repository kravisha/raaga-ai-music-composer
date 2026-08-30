"""Arrangement and orchestration engine (spec sections 7.4, 8).

Adding an instrument never means generating unrelated music.  Every part is
derived from the approved tune: same tonal centre, same raaga, same tempo and
phrase boundaries, with register chosen to stay out of the vocal's way and
density chosen from the section's role.

All edits are region-scoped and non-destructive: a change to 30-45s on the
violin track leaves every other region, track and version untouched, and
refuses outright to touch anything the creator locked.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.models import (ApprovalState, ArrangementVersion, MelodyVersion,
                           Note, Region, Section, SectionKind, Track)
from ..core.versioning import LockedContentError, assert_unlocked_track
from ..raaga.library import Raaga, parse_swara
from .instruments import Instrument, get as get_instrument
from .melody import clamp_token, phrase_boundaries
from .theory import beat_seconds, cycle_seconds, fit_to_range

log = get_logger("arrangement")

ROLES = ("lead", "counter", "pad", "bass", "rhythm", "fill", "drone")

# Stroke patterns as (beat offset, stroke index, velocity) within one cycle.
TALA_PATTERNS: Dict[str, List[Tuple[float, int, int]]] = {
    "adi": [(0, 0, 105), (1, 2, 70), (2, 1, 85), (3, 2, 65),
            (4, 0, 95), (5, 2, 70), (6, 1, 85), (7, 3, 60)],
    "rupaka": [(0, 0, 105), (1, 2, 70), (2, 1, 85), (3, 2, 65), (4, 1, 80), (5, 3, 60)],
    "sparse": [(0, 0, 100), (2, 1, 80), (4, 0, 92), (6, 1, 78)],
    "busy": [(0, 0, 105), (0.5, 3, 55), (1, 2, 70), (1.5, 3, 50), (2, 1, 88),
             (2.5, 3, 55), (3, 2, 66), (3.5, 3, 50), (4, 0, 98), (4.5, 3, 55),
             (5, 2, 70), (5.5, 3, 50), (6, 1, 86), (6.5, 3, 55), (7, 2, 64),
             (7.5, 3, 52)],
}


@dataclass
class PartRequest:
    instrument: str
    role: str = ""
    start: float = 0.0
    end: float = 0.0
    intensity: float = 0.6
    seed: int = 0
    octave_shift: int = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def melody_notes_in(melody: MelodyVersion, start: float, end: float) -> List[Note]:
    return [n for n in melody.notes if n.start < end and start < n.end]


def sections_in(melody: MelodyVersion, start: float, end: float) -> List[Section]:
    return [s for s in melody.sections if s.start < end and start < s.end]


def vocal_register(melody: MelodyVersion) -> Tuple[int, int]:
    sung = [n.midi for n in melody.notes
            if not (melody.section_by_id(n.section_id) or Section()).kind.instrumental]
    if not sung:
        sung = [n.midi for n in melody.notes] or [60]
    return min(sung), max(sung)


def choose_register(inst: Instrument, role: str, melody: MelodyVersion,
                    tonic: int) -> Tuple[int, int]:
    """Pick the instrument's working range so it does not mask the voice."""
    v_low, v_high = vocal_register(melody)
    lo, hi = inst.midi_low, inst.midi_high
    if role == "bass":
        target_low, target_high = tonic - 24, tonic - 5
    elif role == "pad":
        target_low, target_high = tonic - 12, v_low + 2
    elif role == "drone":
        target_low, target_high = tonic - 24, tonic - 2
    elif role == "counter":
        target_low, target_high = v_high - 2, v_high + 14
    elif role == "rhythm":
        target_low, target_high = lo, hi
    else:  # lead / fill
        target_low, target_high = v_low - 2, v_high + 12
    return max(lo, min(target_low, hi - 6)), min(hi, max(target_high, lo + 6))


def suggest_role(inst: Instrument, arrangement: Optional[ArrangementVersion],
                 section: Optional[Section] = None) -> str:
    """Choose a musical job for an instrument given what is already playing."""
    if inst.percussive:
        return "rhythm"
    taken = {t.role for t in (arrangement.tracks if arrangement else [])}
    if section is not None and section.kind.instrumental and inst.supports("lead"):
        return "lead"
    order = [r for r in (inst.default_role, "counter", "pad", "lead", "bass", "fill")
             if inst.supports(r)]
    for role in order:
        if role not in taken:
            return role
    return order[0] if order else inst.default_role


# --------------------------------------------------------------------------
# part writers
# --------------------------------------------------------------------------
def _shift_into(midi: int, low: int, high: int) -> int:
    return fit_to_range(int(midi), int(low), int(high))


def _lead_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
               inst: Instrument, low: int, high: int) -> List[Note]:
    rng = random.Random(req.seed)
    out: List[Note] = []
    for n in melody_notes_in(melody, req.start, req.end):
        start = max(n.start, req.start)
        end = min(n.end, req.end)
        if end - start < 0.04:
            continue
        midi = _shift_into(n.midi + 12 * req.octave_shift, low, high)
        out.append(Note(swara=n.swara, midi=midi, start=round(start, 4),
                        duration=round(end - start, 4),
                        velocity=int(max(35, min(115, n.velocity * (0.6 + 0.6 * req.intensity)))),
                        gamaka=n.gamaka if rng.random() < 0.7 else "",
                        section_id=n.section_id))
    return out


def _counter_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
                  inst: Instrument, low: int, high: int) -> List[Note]:
    """A reply line that fills the gaps the voice leaves."""
    rng = random.Random(req.seed + 17)
    beat = beat_seconds(melody.tempo_bpm)
    out: List[Note] = []
    windows: List[Tuple[float, float, str]] = []
    notes = melody_notes_in(melody, req.start, req.end)
    for i, n in enumerate(notes):
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        gap_start = n.end
        gap_end = nxt.start if nxt else min(req.end, n.end + beat * 2)
        if gap_end - gap_start >= beat * 0.9:
            windows.append((gap_start, min(gap_end, req.end), n.swara))
        # Long held notes get a quiet moving line underneath.
        if n.duration >= beat * 2.2:
            windows.append((n.start + beat, min(n.end, req.end), n.swara))

    for start, end, anchor in windows:
        if end - start < beat * 0.5:
            continue
        token = anchor
        t = start
        while t < end - 0.05:
            dur = min(end - t, beat * rng.choice((0.5, 1.0, 1.0, 1.5)))
            token = raaga.step(token, rng.choice((-2, -1, 1, 1, 2)),
                              1 if rng.random() < 0.5 else -1)
            token = clamp_token(raaga, token, melody.tonic_midi, low, high)
            midi = _shift_into(raaga.midi(token, melody.tonic_midi), low, high)
            out.append(Note(swara=token, midi=midi, start=round(t, 4),
                            duration=round(dur * 0.92, 4),
                            velocity=int(45 + 35 * req.intensity),
                            gamaka=raaga.gamaka_for(token) if rng.random() < 0.4 else "",
                            section_id=_section_id_at(melody, t)))
            t += dur
    return out


def _pad_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
              inst: Instrument, low: int, high: int) -> List[Note]:
    """Sustained stacks that follow the phrase's resting note."""
    out: List[Note] = []
    cyc = cycle_seconds(melody.tempo_bpm, melody.beats_per_cycle)
    t = req.start
    while t < req.end - 0.1:
        end = min(t + cyc, req.end)
        window = melody_notes_in(melody, t, end)
        if window:
            anchor = max(window, key=lambda n: n.duration).swara
        else:
            anchor = raaga.nyasa[0] if raaga.nyasa else "S"
        base = parse_swara(anchor)[0]
        stack = [base]
        for interval in (2, 4):
            stack.append(parse_swara(raaga.step(base, interval, 1))[0])
        for k, sw in enumerate(stack):
            token = clamp_token(raaga, sw, melody.tonic_midi, low, high)
            midi = _shift_into(raaga.midi(token, melody.tonic_midi) + 12 * (k // 3),
                               low, high)
            out.append(Note(swara=token, midi=midi, start=round(t, 4),
                            duration=round(end - t, 4),
                            velocity=int(34 + 26 * req.intensity),
                            section_id=_section_id_at(melody, t)))
        t = end
    return out


def _bass_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
               inst: Instrument, low: int, high: int) -> List[Note]:
    out: List[Note] = []
    beat = beat_seconds(melody.tempo_bpm)
    cyc = cycle_seconds(melody.tempo_bpm, melody.beats_per_cycle)
    rng = random.Random(req.seed + 31)
    t = req.start
    while t < req.end - 0.05:
        end = min(t + cyc / 2, req.end)
        window = melody_notes_in(melody, t, end)
        anchor = (max(window, key=lambda n: n.duration).swara if window
                  else (raaga.nyasa[0] if raaga.nyasa else "S"))
        base = parse_swara(anchor)[0]
        if base not in ("S", "P", "M1", "M2") and rng.random() < 0.6:
            base = "S" if rng.random() < 0.6 else ("P" if "P" in raaga.allowed else "S")
        token = clamp_token(raaga, base, melody.tonic_midi, low, high)
        midi = _shift_into(raaga.midi(token, melody.tonic_midi), low, high)
        dur = min(end - t, beat * 1.6)
        out.append(Note(swara=token, midi=midi, start=round(t, 4),
                        duration=round(dur, 4),
                        velocity=int(55 + 35 * req.intensity),
                        section_id=_section_id_at(melody, t)))
        t = end
    return out


def _drone_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
                inst: Instrument, low: int, high: int) -> List[Note]:
    out: List[Note] = []
    cyc = cycle_seconds(melody.tempo_bpm, melody.beats_per_cycle)
    fifth = "P" if "P" in raaga.allowed else ("M1" if "M1" in raaga.allowed else "S")
    t = req.start
    while t < req.end - 0.05:
        end = min(t + cyc, req.end)
        for sw in ("S", fifth, "S"):
            token = clamp_token(raaga, sw, melody.tonic_midi, low, high)
            midi = _shift_into(raaga.midi(token, melody.tonic_midi), low, high)
            out.append(Note(swara=token, midi=midi, start=round(t, 4),
                            duration=round(end - t, 4),
                            velocity=int(30 + 18 * req.intensity),
                            section_id=_section_id_at(melody, t)))
        t = end
    return out


def _rhythm_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
                 inst: Instrument, low: int, high: int) -> List[Note]:
    beat = beat_seconds(melody.tempo_bpm)
    cyc = cycle_seconds(melody.tempo_bpm, melody.beats_per_cycle)
    if req.intensity > 0.75:
        pattern = TALA_PATTERNS["busy"]
    elif req.intensity < 0.4:
        pattern = TALA_PATTERNS["sparse"]
    else:
        pattern = TALA_PATTERNS["adi" if melody.beats_per_cycle % 8 == 0
                                else "rupaka"]
    out: List[Note] = []
    t = req.start
    while t < req.end - 0.05:
        for offset, stroke, vel in pattern:
            at = t + offset * beat
            if at >= req.end or at < req.start:
                continue
            out.append(Note(swara="-", midi=36 + stroke, start=round(at, 4),
                            duration=round(beat * 0.4, 4),
                            velocity=int(vel * (0.55 + 0.6 * req.intensity)),
                            section_id=_section_id_at(melody, at)))
        t += cyc
    return out


def _fill_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest,
               inst: Instrument, low: int, high: int) -> List[Note]:
    """Short flourishes at section joins and long rests."""
    rng = random.Random(req.seed + 53)
    beat = beat_seconds(melody.tempo_bpm)
    out: List[Note] = []
    anchors: List[float] = []
    for s in sections_in(melody, req.start, req.end):
        if s.start > req.start + 0.1:
            anchors.append(max(req.start, s.start - beat * 2))
    for group in phrase_boundaries(melody):
        last = melody.notes[group[-1]]
        if req.start <= last.end < req.end:
            anchors.append(last.end)
    for at in sorted(set(round(a, 2) for a in anchors)):
        token = raaga.nyasa[0] if raaga.nyasa else "S"
        t = at
        frag = list(rng.choice(raaga.prayogas)) if raaga.prayogas else ["S", "R2", "G2"]
        for sw in frag:
            if t >= req.end:
                break
            token = clamp_token(raaga, sw, melody.tonic_midi, low, high)
            midi = _shift_into(raaga.midi(token, melody.tonic_midi), low, high)
            dur = beat * 0.5
            out.append(Note(swara=token, midi=midi, start=round(t, 4),
                            duration=round(dur * 0.9, 4),
                            velocity=int(50 + 35 * req.intensity),
                            gamaka=raaga.gamaka_for(token),
                            section_id=_section_id_at(melody, t)))
            t += dur
    return out


PART_WRITERS = {
    "lead": _lead_part,
    "counter": _counter_part,
    "pad": _pad_part,
    "bass": _bass_part,
    "rhythm": _rhythm_part,
    "fill": _fill_part,
    "drone": _drone_part,
}


def _section_id_at(melody: MelodyVersion, t: float) -> str:
    s = melody.section_at(t)
    return s.id if s else ""


def generate_part(melody: MelodyVersion, raaga: Raaga, req: PartRequest
                  ) -> List[Note]:
    inst = get_instrument(req.instrument)
    if inst is None:
        raise KeyError(f"Unknown instrument: {req.instrument}")
    role = req.role or inst.default_role
    writer = PART_WRITERS.get(role)
    if writer is None:
        raise ValueError(f"Unknown role: {role}")
    low, high = choose_register(inst, role, melody, melody.tonic_midi)
    notes = writer(melody, raaga, req, inst, low, high)
    return sorted(notes, key=lambda n: n.start)


# --------------------------------------------------------------------------
# arrangement operations
# --------------------------------------------------------------------------
def new_version(previous: Optional[ArrangementVersion], label: str = ""
                ) -> ArrangementVersion:
    version = (previous.version + 1) if previous else 1
    arrangement = ArrangementVersion(version=version, label=label or f"Arrangement v{version}")
    if previous:
        arrangement.tracks = deepcopy(previous.tracks)
    return arrangement


def find_track(arrangement: ArrangementVersion, instrument: str,
               role: str = "") -> Optional[Track]:
    for t in arrangement.tracks:
        if t.instrument == instrument and (not role or t.role == role):
            return t
    return None


def add_instrument(arrangement: ArrangementVersion, melody: MelodyVersion,
                   raaga: Raaga, instrument: str, start: float, end: float,
                   role: str = "", intensity: float = 0.6, seed: int = 0,
                   octave_shift: int = 0, generated_by: str = "creator"
                   ) -> Tuple[Track, Region]:
    """Add (or extend) an instrument over a time range."""
    inst = get_instrument(instrument)
    if inst is None:
        raise KeyError(instrument)
    section = melody.section_at(start)
    role = role or suggest_role(inst, arrangement, section)
    if not inst.supports(role):
        role = inst.default_role

    track = find_track(arrangement, instrument, role)
    if track is None:
        track = Track(instrument=instrument, role=role,
                      display_name=inst.name, gain=inst.default_gain,
                      pan=inst.default_pan, created_by=generated_by)
        arrangement.tracks.append(track)
    else:
        assert_unlocked_track(track, start, end, f"{inst.name} {start:.0f}-{end:.0f}s")

    req = PartRequest(instrument=instrument, role=role, start=start, end=end,
                      intensity=intensity, seed=seed or int(start * 1000) + 7,
                      octave_shift=octave_shift)
    notes = generate_part(melody, raaga, req)
    region = Region(start=start, end=end, role=role, notes=notes,
                    seed=req.seed, generated_by=generated_by,
                    meta={"intensity": f"{intensity:.2f}"})
    # Trim any unlocked overlap so regions never double up.
    keep: List[Region] = []
    for r in track.regions:
        if r.locked or not r.overlaps(start, end):
            keep.append(r)
    track.regions = keep + [region]
    track.regions.sort(key=lambda r: r.start)
    log.info("added %s (%s) %.1f-%.1fs with %d notes", instrument, role, start, end,
             len(notes))
    return track, region


def remove_instrument(arrangement: ArrangementVersion, instrument: str,
                      start: Optional[float] = None, end: Optional[float] = None
                      ) -> int:
    """Remove an instrument entirely, or only within a time range."""
    removed = 0
    for track in list(arrangement.tracks):
        if track.instrument != instrument:
            continue
        if start is None or end is None:
            if track.locked or any(r.locked for r in track.regions):
                raise LockedContentError(
                    f"'{track.label}' has locked content; unlock it first.")
            removed += len(track.regions)
            arrangement.tracks.remove(track)
        else:
            assert_unlocked_track(track, start, end, "that range")
            before = len(track.regions)
            track.regions = [r for r in track.regions if not r.overlaps(start, end)]
            removed += before - len(track.regions)
            if not track.regions:
                arrangement.tracks.remove(track)
    return removed


def replace_instrument(arrangement: ArrangementVersion, melody: MelodyVersion,
                       raaga: Raaga, old_instrument: str, new_instrument: str,
                       start: Optional[float] = None, end: Optional[float] = None
                       ) -> List[Tuple[Track, Region]]:
    """Swap one instrument for another over the same time ranges and roles."""
    if get_instrument(new_instrument) is None:
        raise KeyError(new_instrument)
    targets: List[Tuple[str, float, float, float, int]] = []
    for track in arrangement.tracks:
        if track.instrument != old_instrument:
            continue
        for region in track.regions:
            if start is not None and end is not None and not region.overlaps(start, end):
                continue
            if region.locked or track.locked:
                raise LockedContentError(
                    f"'{track.label}' {region.start:.0f}-{region.end:.0f}s is locked.")
            lo = max(region.start, start) if start is not None else region.start
            hi = min(region.end, end) if end is not None else region.end
            targets.append((track.role, lo, hi,
                            float(region.meta.get("intensity", 0.6)), region.seed))
    if not targets:
        raise LookupError(f"{old_instrument} is not playing there.")

    remove_instrument(arrangement, old_instrument, start, end)
    out = []
    for role, lo, hi, intensity, seed in targets:
        out.append(add_instrument(arrangement, melody, raaga, new_instrument, lo, hi,
                                  role=role, intensity=intensity, seed=seed,
                                  generated_by="replace"))
    return out


def regenerate_region(arrangement: ArrangementVersion, melody: MelodyVersion,
                      raaga: Raaga, track_id: str, region_id: str,
                      seed: Optional[int] = None) -> Region:
    track = arrangement.track_by_id(track_id)
    if track is None:
        raise KeyError(track_id)
    region = track.region_by_id(region_id)
    if region is None:
        raise KeyError(region_id)
    if track.locked or region.locked:
        raise LockedContentError(f"'{track.label}' region is locked.")
    req = PartRequest(instrument=track.instrument, role=region.role,
                      start=region.start, end=region.end,
                      intensity=float(region.meta.get("intensity", 0.6)),
                      seed=seed if seed is not None else region.seed + 101)
    region.notes = generate_part(melody, raaga, req)
    region.seed = req.seed
    region.version += 1
    region.generated_by = "regenerate"
    return region


def move_region(arrangement: ArrangementVersion, melody: MelodyVersion,
                raaga: Raaga, track_id: str, region_id: str,
                new_start: float, new_end: float) -> Region:
    track = arrangement.track_by_id(track_id)
    if track is None:
        raise KeyError(track_id)
    region = track.region_by_id(region_id)
    if region is None:
        raise KeyError(region_id)
    if track.locked or region.locked:
        raise LockedContentError(f"'{track.label}' region is locked.")
    req = PartRequest(instrument=track.instrument, role=region.role,
                      start=new_start, end=new_end,
                      intensity=float(region.meta.get("intensity", 0.6)),
                      seed=region.seed)
    region.start, region.end = new_start, new_end
    region.notes = generate_part(melody, raaga, req)
    region.version += 1
    return region


def set_region_lock(arrangement: ArrangementVersion, track_id: str,
                    region_id: str, locked: bool) -> Region:
    track = arrangement.track_by_id(track_id)
    if track is None:
        raise KeyError(track_id)
    region = track.region_by_id(region_id)
    if region is None:
        raise KeyError(region_id)
    region.locked = locked
    return region


def lock_range(arrangement: ArrangementVersion, start: float, end: float,
               locked: bool = True) -> int:
    n = 0
    for track in arrangement.tracks:
        for region in track.regions:
            if region.overlaps(start, end):
                region.locked = locked
                n += 1
    return n


# --------------------------------------------------------------------------
# automatic first pass
# --------------------------------------------------------------------------
def auto_arrange(melody: MelodyVersion, raaga: Raaga, brief, seed: int = 5,
                 previous: Optional[ArrangementVersion] = None
                 ) -> ArrangementVersion:
    """Build a complete, playable first arrangement from the tune."""
    from .instruments import find as find_instrument, suggest_for_feel
    from ..raaga.selection import expand_feel_words

    arrangement = new_version(previous, label="Auto arrangement")
    total = melody.duration
    words = expand_feel_words(brief.mood, brief.feel, brief.situation, brief.notes)

    preferred = [i for i in (find_instrument(p) for p in brief.instruments_preferred)
                 if i is not None]
    avoid = list(brief.instruments_avoided)

    lead_candidates = [i for i in preferred if i.supports("lead")]
    if not lead_candidates:
        ranked = suggest_for_feel(words, avoid, role="lead", limit=3)
        lead_candidates = [i for i, _ in ranked] or [find_instrument("flute")]
    lead = lead_candidates[0]

    pad_ranked = suggest_for_feel(words, avoid + [lead.key], role="pad", limit=2)
    pad = pad_ranked[0][0] if pad_ranked else find_instrument("strings")

    percussion = next((i for i in preferred if i.percussive), None)
    if percussion is None:
        percussion = find_instrument("mridangam" if "carnatic" in " ".join(words)
                                     or brief.language.lower() in ("tamil", "telugu")
                                     else "tabla")

    drone = find_instrument("tanpura")
    bass = find_instrument("double_bass" if "night" in words else "bass")

    if drone:
        add_instrument(arrangement, melody, raaga, drone.key, 0.0, total,
                       role="drone", intensity=0.4, seed=seed,
                       generated_by="auto")
    if pad:
        add_instrument(arrangement, melody, raaga, pad.key, 0.0, total,
                       role="pad", intensity=0.5, seed=seed + 1,
                       generated_by="auto")
    if bass:
        add_instrument(arrangement, melody, raaga, bass.key, 0.0, total,
                       role="bass", intensity=0.55, seed=seed + 2,
                       generated_by="auto")
    if percussion:
        # Percussion enters after the prelude.
        first_sung = next((s.start for s in melody.sections
                           if not s.kind.instrumental), 0.0)
        add_instrument(arrangement, melody, raaga, percussion.key, first_sung, total,
                       role="rhythm", intensity=0.6, seed=seed + 3,
                       generated_by="auto")
    if lead:
        for section in melody.sections:
            if section.kind.instrumental:
                add_instrument(arrangement, melody, raaga, lead.key,
                               section.start, section.end, role="lead",
                               intensity=section.intensity, seed=seed + 4,
                               generated_by="auto")
            elif section.kind in (SectionKind.PALLAVI, SectionKind.CHORUS):
                add_instrument(arrangement, melody, raaga, lead.key,
                               section.start, section.end, role="counter",
                               intensity=0.45, seed=seed + 5,
                               generated_by="auto")
    return arrangement


def describe(arrangement: ArrangementVersion) -> str:
    if not arrangement.tracks:
        return "(no instruments yet)"
    rows = []
    for t in arrangement.tracks:
        spans = ", ".join(f"{r.start:.0f}-{r.end:.0f}s{'*' if r.locked else ''}"
                          for r in t.regions)
        flags = ("M" if t.mute else "") + ("S" if t.solo else "")
        rows.append(f"{t.label:<18} {t.role:<8} {spans} {flags}"
                    f"{' [locked]' if t.locked else ''}")
    return "\n".join(rows)
