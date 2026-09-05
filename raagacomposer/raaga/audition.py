"""Hearing a raaga's scale - Stage 1 pack document 05 section 7, test E.

The pack's audition step: the creator picks a raaga from the suggestions and
hears it before anything is composed.  ``Play Arohana.  Play Avarohana.  User
confirms swara/raga identity.``  A ranked list with a reason attached is an
argument; the scale played is the evidence.

What the pack requires of the playback (document 01 section H, document 06
test E) is small and exact, and all of it is checkable without audio, which
is why the plan is built here as data and rendered elsewhere:

* the arohanam then the avarohanam, as stored, in that order;
* every note keeps its functional swara label - "never store only MIDI note
  number", because the label is what makes the audition a *swara* test rather
  than a tune;
* pitches follow the stored pitch classes;
* and it must never collapse into one repeated note, which is the failure
  the pack names twice and the one an untested implementation actually
  produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from ..core.models import Note
from .library import Raaga

#: Middle C.  The same tonic the practice engine hears in, so an audition and
#: the agent's own exercises are the same pitch and can be compared by ear.
TONIC = 60

#: Slow enough to hear each swara as itself rather than as part of a run.  An
#: audition is not a performance: the creator is being asked "is this the
#: raaga you meant", and that question needs the notes separated.
NOTE_SECONDS = 0.55
GAP_SECONDS = 0.12
#: A breath between the way up and the way down, so the two directions are
#: audibly two phrases rather than one sixteen-note scale.
TURN_SECONDS = 0.35


@dataclass
class AuditionPlan:
    """What to play, as swaras and pitches, before any audio exists."""

    raaga: str = ""
    ascending: List[Note] = field(default_factory=list)
    descending: List[Note] = field(default_factory=list)

    @property
    def notes(self) -> List[Note]:
        return list(self.ascending) + list(self.descending)

    @property
    def seconds(self) -> float:
        last = self.notes[-1] if self.notes else None
        return (last.start + last.duration + 0.3) if last else 0.0

    @property
    def pitches(self) -> List[int]:
        return [n.midi for n in self.notes]

    @property
    def swaras(self) -> List[str]:
        return [n.swara for n in self.notes]

    def describe(self) -> str:
        return (f"{self.raaga}: "
                f"{' '.join(n.swara for n in self.ascending)}  /  "
                f"{' '.join(n.swara for n in self.descending)}")


def _line(swaras: Sequence[str], raaga: Raaga, tonic: int,
          start: float) -> List[Note]:
    notes: List[Note] = []
    at = start
    for swara in swaras:
        notes.append(Note(swara=swara, midi=raaga.midi(swara, tonic),
                          start=at, duration=NOTE_SECONDS, velocity=92))
        at += NOTE_SECONDS + GAP_SECONDS
    return notes


def plan(raaga: Raaga, tonic: int = TONIC) -> AuditionPlan:
    """The exact arohanam and avarohanam of this raaga, ready to render.

    Exact is the point.  This does not generate a phrase, choose a register
    or ornament anything: it plays what the library stores, so that what the
    creator hears is the raaga's own grammar and a disagreement is about the
    raaga rather than about the performance.
    """
    ascending = _line(raaga.arohanam, raaga, tonic, 0.0)
    after = (ascending[-1].start + ascending[-1].duration + TURN_SECONDS
             if ascending else 0.0)
    descending = _line(raaga.avarohanam, raaga, tonic, after)
    return AuditionPlan(raaga=raaga.name, ascending=ascending,
                        descending=descending)


def is_playable(plan: AuditionPlan) -> bool:
    """Does this actually sound like a scale, rather than one note repeated?

    Pack document 01 section H rule 7 and document 06 test E, checked rather
    than assumed: both directions present, and the pitches genuinely move.
    """
    if not plan.ascending or not plan.descending:
        return False
    return len(set(plan.pitches)) > 2
