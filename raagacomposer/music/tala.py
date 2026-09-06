"""Tala: the rhythmic cycle, as a thing the application knows about.

Rhythm existed here only as ``beats_per_cycle``, an integer, and a handful
of stroke patterns keyed by the words "adi" and "rupaka".  A tala is not a
number of beats.  It is a named cycle with an internal shape - Adi is
8 beats as 4+2+2, Misra Chapu is 7 as 3+2+2 - and that shape is what a
percussionist plays to, what a phrase lands on, and what a listener feels.

Raga and tala are the two first-class dimensions of this music
(specification 6.4), and only one of them was represented.

What each tala carries:

``aksharas``    beats in one cycle
``angas``       the cycle's internal grouping, which is where the accents
                fall and where a phrase may land
``accents``     beats that carry weight, derived from the angas rather than
                asserted separately, so the two can never disagree

Stroke patterns live here too, expressed in beats from the start of the
cycle so they hold at any tempo and in any nadai.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: A stroke: (beat offset within the cycle, stroke index, velocity 0-127).
#: The stroke index selects a sound from the instrument's ``hit_freqs`` -
#: 0 is the deepest, higher indices are brighter and lighter.
Stroke = Tuple[float, int, int]


@dataclass(frozen=True)
class Tala:
    """A rhythmic cycle and the shape inside it."""

    name: str
    aksharas: int
    angas: Tuple[int, ...]
    aliases: Tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if sum(self.angas) != self.aksharas:
            raise ValueError(
                f"{self.name}: angas {self.angas} sum to {sum(self.angas)}, "
                f"not {self.aksharas} aksharas")

    @property
    def accents(self) -> Tuple[int, ...]:
        """The beat each anga starts on: where the weight falls."""
        out, beat = [], 0
        for length in self.angas:
            out.append(beat)
            beat += length
        return tuple(out)

    def is_accent(self, beat: float) -> bool:
        return int(beat) % self.aksharas in self.accents

    def describe(self) -> str:
        shape = "+".join(str(a) for a in self.angas)
        return f"{self.name}, {self.aksharas} beats ({shape})"


#: The talas a Carnatic composer reaches for first.  Deliberately short:
#: a small set that is right beats a long one that is half-checked.
TALAS: Tuple[Tala, ...] = (
    Tala("Adi", 8, (4, 2, 2), ("adi", "adi tala", "chaturasra triputa"),
         "The most common tala; 8 beats as 4+2+2."),
    Tala("Rupaka", 6, (2, 4), ("rupaka", "rupakam"),
         "Six beats as 2+4, brisk and song-like."),
    Tala("Misra Chapu", 7, (3, 2, 2), ("misra chapu", "misra", "chapu"),
         "Seven beats as 3+2+2; lilting and asymmetric."),
    Tala("Khanda Chapu", 5, (2, 3), ("khanda chapu", "khanda"),
         "Five beats as 2+3; short and driving."),
    Tala("Tisra Eka", 3, (3,), ("tisra eka", "tisra"),
         "Three beats to a cycle; waltz-like."),
)

DEFAULT_TALA = "Adi"


def find(name: str) -> Optional[Tala]:
    """Look a tala up by name or alias.  Returns None rather than guessing."""
    key = " ".join(str(name or "").lower().split())
    if not key:
        return None
    for tala in TALAS:
        if key == tala.name.lower() or key in tala.aliases:
            return tala
    return None


def require(name: str) -> Tala:
    tala = find(name) or find(DEFAULT_TALA)
    assert tala is not None                     # DEFAULT_TALA is in TALAS
    return tala


def for_beats(aksharas: int) -> Tala:
    """The tala with this many beats, for melodies that only carry a count.

    A melody records ``beats_per_cycle`` and nothing else, so this is how
    an existing tune says which cycle it was written in.
    """
    for tala in TALAS:
        if tala.aksharas == aksharas:
            return tala
    return require(DEFAULT_TALA)


def names() -> List[str]:
    return [t.name for t in TALAS]


# --------------------------------------------------------------------------
# stroke patterns
# --------------------------------------------------------------------------
#: Density names, quietest first.  Density is *how much* is played; the
#: tala decides *where* the weight falls, so the two are independent.
DENSITIES = ("sparse", "steady", "busy")
DEFAULT_DENSITY = "steady"


def pattern(tala: Tala, density: str = DEFAULT_DENSITY) -> List[Stroke]:
    """One cycle of strokes for this tala at this density.

    Built from the tala's own shape rather than written out per tala, so a
    tala added above gets a musical pattern without a table entry: the
    first beat of each anga is accented, the beats between are lighter, and
    "busy" fills the half-beats.
    """
    density = density if density in DENSITIES else DEFAULT_DENSITY
    accents = set(tala.accents)
    strokes: List[Stroke] = []
    for beat in range(tala.aksharas):
        if beat in accents:
            # The cycle's first beat is the heaviest of the accents.
            strokes.append((float(beat), 0, 105 if beat == 0 else 88))
        elif density == "sparse":
            continue
        else:
            strokes.append((float(beat), 1 if beat % 2 else 2, 70))
    if density == "busy":
        for beat in range(tala.aksharas):
            strokes.append((beat + 0.5, 3, 52))
    return sorted(strokes)
