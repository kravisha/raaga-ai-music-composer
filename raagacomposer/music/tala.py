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

import re
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


#: Words that point at a cycle, and why.  Each cycle is chosen for what
#: it does rather than by frequency: five beats drive, seven lilt, three
#: sway, six move briskly.  Longest phrase first, so "slow waltz" is read
#: as a waltz rather than as slow.
_POINTERS: Tuple[Tuple[str, str, str], ...] = (
    ("khanda chapu", "Khanda Chapu", "you named it"),
    ("misra chapu", "Misra Chapu", "you named it"),
    ("tisra eka", "Tisra Eka", "you named it"),
    ("chase", "Khanda Chapu", "a chase often wants a short driving cycle"),
    ("urgent", "Khanda Chapu", "urgency often wants a short driving cycle"),
    ("frantic", "Khanda Chapu", "frantic often wants a short driving cycle"),
    ("restless", "Khanda Chapu", "restlessness often wants a short driving cycle"),
    ("nervous", "Khanda Chapu", "nervousness often wants a short driving cycle"),
    ("waltz", "Tisra Eka", "a waltz is three beats to a cycle"),
    ("lullaby", "Tisra Eka", "a lullaby often sways in three"),
    ("cradle", "Tisra Eka", "a cradle song often sways in three"),
    ("sway", "Tisra Eka", "swaying often sits in three"),
    ("folk", "Misra Chapu", "folk songs often lilt in seven"),
    ("lilting", "Misra Chapu", "a lilt is the seven-beat cycle"),
    ("playful", "Misra Chapu", "playfulness often suits the asymmetric seven"),
    ("skipping", "Misra Chapu", "skipping often suits the asymmetric seven"),
    ("brisk", "Rupaka", "brisk and song-like is the six-beat cycle"),
    ("light", "Rupaka", "a light song often sits in six"),
    ("marching", "Adi", "a march often wants the even eight"),
    ("stately", "Adi", "stateliness often wants the even eight"),
    ("devotional", "Adi", "devotional songs often sit in the eight-beat cycle"),
)


@dataclass(frozen=True)
class TalaChoice:
    """A cycle, and why it was chosen - so the creator can disagree."""

    tala: Tala
    reason: str
    chosen_by_creator: bool = False

    def describe(self) -> str:
        if self.chosen_by_creator:
            return f"you chose {self.tala.describe()} - {self.reason}"
        # Said as a suggestion, because it is one.  No lullaby has to be in
        # three and no folk song in seven; these are starting points read
        # off the words, and the picker beside this changes them.
        return (f"I suggest {self.tala.describe()} - {self.reason}. "
                f"Change it with the picker if it is not what you hear.")


def suggest(brief, raaga=None) -> TalaChoice:
    """Pick a cycle for a new tune, and say why.

    A default is not a choice.  Every new tune used to be composed in Adi
    because Adi is what ``require`` falls back to, and nothing told the
    creator that a decision had been made on their behalf or offered them
    another.  This reads the brief the way infer_tempo reads it for speed.
    """
    named = find(getattr(brief, "tala", ""))
    if named is not None:
        return TalaChoice(named, "you asked for this cycle", True)

    # Read the most specific statement first.  A named scene beats a mood
    # adjective, and the default mood - "tense, upbeat, nervous, excited,
    # hopeful" - is on every new brief, so scanning one merged blob meant
    # every song in the application suggested the same driving cycle
    # whatever it was about.
    for field in ("situation", "feel", "notes", "song_type", "mood"):
        blob = str(getattr(brief, field, "") or "").lower()
        if not blob:
            continue
        choice = _read(blob, field)
        if choice is not None:
            return choice
    return TalaChoice(require(DEFAULT_TALA),
                      "nothing in the brief pointed elsewhere, and Adi is "
                      "where a Carnatic song starts")


def _read(blob: str, field: str) -> Optional["TalaChoice"]:
    """The first cycle this one piece of the brief points at, if any."""
    for phrase, tala_name, why in _POINTERS:
        if not _mentions(blob, phrase):
            continue
        if _rejected(blob, phrase):
            # "a lullaby, not a chase" names a chase in order to rule it
            # out.  Reading that as a request for one, and then citing it
            # as the reason, is worse than not reading it at all.
            continue
        found = find(tala_name)
        if found is not None:
            where = "the situation" if field == "situation" else f"the {field}"
            return TalaChoice(found, f"{why} (\"{phrase}\" in {where})")
    return None


#: Words that turn the phrase after them into something the creator does
#: not want.  Kept short and near: "not" three words back is a rejection,
#: "not" a sentence back is usually about something else.
_NEGATORS = ("not", "no", "never", "without", "avoid", "instead of",
             "rather than", "isn't", "is not", "anything but")


def _mentions(blob: str, phrase: str) -> bool:
    """Whole words only.

    "chase" is inside "purchase", and a brief about a quiet purchase of a
    gift was read as a chase.  The library's voice lookup already had this
    fixed for the same reason - "male" must not match "Female - Warm" -
    and I did not carry it across.
    """
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])",
                     blob) is not None


def _rejected(blob: str, phrase: str) -> bool:
    """Whether the brief names this idea only to turn it down."""
    for match in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", blob):
        before = blob[max(0, match.start() - 40):match.start()]
        tail = re.split(r"[.;]", before)[-1]
        words = re.findall(r"[a-z']+", tail)[-4:]
        window = " ".join(words)
        if any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", window)
               for n in _NEGATORS):
            continue                    # this mention is a rejection
        return False                    # some mention is a genuine request
    return True


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
