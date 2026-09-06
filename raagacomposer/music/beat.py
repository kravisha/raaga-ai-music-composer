"""The beat: percussion you can make, hear and vary on its own.

Percussion existed only as a side effect of arranging the whole song.  A
creator could not make a beat, could not hear one without a full mix, and
could not change one without rebuilding the arrangement around it.  The
specification asks for percussion as a first-class musical layer
(section 11) - generated separately, varied separately, and mixed with the
melody rather than baked into it.

The beat is written against the *tala*, not against the melody's notes, so
regenerating it cannot disturb a tune (11.9) and a tune can be replaced
without losing the groove.  They meet at the cycle: same tempo, same
aksharas, so they line up by construction rather than by alignment.

Variation preserves the concept (11.2).  A variation of a beat is
recognisably the same beat - the accents of the tala are never moved,
because moving them makes it a different tala rather than a variation -
and what changes is ornament: the strokes between the accents, the fills
at cycle ends, the weight of the hand.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ..core.models import BeatVersion, Note
from .tala import DEFAULT_DENSITY, DENSITIES, Stroke, Tala, pattern, require
from .theory import beat_seconds

#: How far a variation may move.  "Subtle" changes ornament only; "twist"
#: may change the density itself.  The accents never move at any strength.
STRENGTHS = ("subtle", "moderate", "twist")
DEFAULT_STRENGTH = "moderate"

#: Percussion strokes are pitched by index into the instrument's hit
#: sounds, not by scale degree, so the "midi" of a stroke is a selector.
STROKE_BASE_MIDI = 36


def _stroke_note(offset_beats: float, stroke: int, velocity: int,
                 start: float, beat: float) -> Note:
    at = start + offset_beats * beat
    return Note(swara="-", midi=STROKE_BASE_MIDI + stroke,
                start=round(at, 4), duration=round(beat * 0.4, 4),
                velocity=max(1, min(127, velocity)))


def generate(tala: Tala, tempo_bpm: int, duration: float, *,
             density: str = DEFAULT_DENSITY, seed: int = 0,
             intensity: float = 0.6) -> List[Note]:
    """Fill ``duration`` seconds with cycles of ``tala``."""
    beat = beat_seconds(tempo_bpm)
    cycle = beat * tala.aksharas
    rng = random.Random(seed)
    strokes: Sequence[Stroke] = pattern(tala, density)

    notes: List[Note] = []
    t = 0.0
    cycle_index = 0
    while t < duration - 0.05:
        last_cycle = (t + cycle) >= (duration - 0.05)
        for offset, stroke, velocity in strokes:
            if t + offset * beat >= duration:
                continue
            # A hand is not a machine: a few percent of weight either way.
            shaped = int(velocity * (0.55 + 0.6 * intensity)
                         * rng.uniform(0.94, 1.06))
            notes.append(_stroke_note(offset, stroke, shaped, t, beat))
        # A fill lands into the next cycle's first beat, which is what makes
        # a cycle boundary audible instead of merely arithmetic.
        if not last_cycle and cycle_index % 4 == 3:
            for i, off in enumerate((tala.aksharas - 0.75, tala.aksharas - 0.5,
                                     tala.aksharas - 0.25)):
                notes.append(_stroke_note(off, 3, 46 + 6 * i, t, beat))
        t += cycle
        cycle_index += 1
    notes.sort(key=lambda n: n.start)
    return notes


def vary(previous: BeatVersion, *, strength: str = DEFAULT_STRENGTH,
         seed: Optional[int] = None) -> BeatVersion:
    """A different take on the same beat, not a different beat.

    The tala and its accents survive every strength, because a beat whose
    accents have moved is not a variation of this one.
    """
    strength = strength if strength in STRENGTHS else DEFAULT_STRENGTH
    tala = require(previous.tala)
    rng = random.Random(seed if seed is not None else previous.seed + 17)

    density = previous.density
    intensity = previous.intensity
    if strength == "moderate":
        intensity = min(1.0, max(0.15, intensity + rng.uniform(-0.15, 0.15)))
    elif strength == "twist":
        # Only a twist may change how much is played; subtler strengths
        # would no longer be recognisable as the same beat.
        options = [d for d in DENSITIES if d != density] or list(DENSITIES)
        density = rng.choice(options)
        intensity = min(1.0, max(0.15, intensity + rng.uniform(-0.25, 0.25)))

    return BeatVersion(
        version=previous.version + 1,
        label=f"{strength.title()} variation",
        tala=tala.name, tempo_bpm=previous.tempo_bpm,
        density=density, intensity=round(intensity, 3),
        seed=rng.randint(1, 10_000), duration=previous.duration,
        parent_version=previous.version, strength=strength,
        notes=[])


def realise(version: BeatVersion) -> BeatVersion:
    """Fill in a version's notes from its own settings."""
    version.notes = generate(
        require(version.tala), version.tempo_bpm, version.duration,
        density=version.density, seed=version.seed,
        intensity=version.intensity)
    return version
