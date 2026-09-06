"""Who plays a part, and why - one decision, made in one place.

Two paths chose a lead instrument and chose it differently.  The audition
read the brief's preferred instruments, then a configured default, then
fell back to the veena.  The arrangement read the brief's preferred
instruments, then ranked the whole catalogue against the *feel* of the
brief, then fell back to the flute.

So a creator could hear a raaga auditioned on one instrument and the
arrangement built on another, from the same brief, with nothing on screen
explaining the difference.  Neither policy was wrong; having two of them
was.

This module is the single answer.  Both paths call it, so they cannot
drift apart again without the drift being visible here.

The decision carries its reason.  "Which instrument" and "why that one"
are the same question asked twice, and the interface has to answer the
second - a creator told the application to prefer a violin deserves to
see that the violin is why, and a creator who did not deserves to see
what stood in for their silence.

The shape generalises: a *role* to fill, an ordered set of *preferences*,
constraints to respect, and a default that is never silent about being a
default.  Nothing here is specific to melody - the same order applies to
any part that has to be cast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .instruments import Instrument, find, suggest_for_feel

#: Where a choice came from, in the order they are tried.
FROM_BRIEF = "brief"          # the creator named it
FROM_SETTING = "setting"      # a configured preference
FROM_FEEL = "feel"            # ranked against what the brief feels like
FROM_DEFAULT = "default"      # nothing else applied


@dataclass(frozen=True)
class Casting:
    """One filled part: the instrument, where the choice came from, and why."""

    instrument: Instrument
    source: str
    reason: str

    @property
    def chosen_by_the_creator(self) -> bool:
        return self.source == FROM_BRIEF

    def describe(self) -> str:
        return f"{self.instrument.name} - {self.reason}"


def cast(role: str = "lead", *,
         preferred: Sequence[str] = (),
         avoided: Sequence[str] = (),
         feel_words: Sequence[str] = (),
         configured: str = "",
         default: str = "veena",
         rank: Optional[Callable[..., List]] = None) -> Casting:
    """Choose the instrument for ``role``, and say why.

    The order, most deliberate first:

    1. an instrument the creator asked for that can carry the role;
    2. a configured preference that can carry the role;
    3. the catalogue ranked against what the brief feels like;
    4. the stated default.

    ``rank`` exists so a caller can supply a richer ranking - the
    controller's, which asks a language model before falling back to the
    lexicon - without this module knowing anything about providers.
    """
    avoid = {str(a).strip().lower() for a in avoided if str(a).strip()}

    def usable(instrument: Optional[Instrument]) -> bool:
        return (instrument is not None
                and instrument.supports(role)
                and instrument.key.lower() not in avoid
                and instrument.name.lower() not in avoid)

    for name in preferred:
        instrument = find(str(name).strip())
        if usable(instrument):
            return Casting(instrument, FROM_BRIEF,
                           f"you asked for {instrument.name.lower()}")

    configured_instrument = find(str(configured).strip()) if configured else None
    if usable(configured_instrument):
        return Casting(configured_instrument, FROM_SETTING,
                       f"your configured {role} instrument")

    words = [w for w in feel_words if w]
    if words:
        ranker = rank or suggest_for_feel
        try:
            ranked = ranker(words, list(avoid), role=role, limit=3)
        except Exception:                       # noqa: BLE001 - never fatal
            ranked = []
        for instrument, _score in ranked:
            if usable(instrument):
                return Casting(
                    instrument, FROM_FEEL,
                    f"it suits {', '.join(words[:3])}")

    fallback = find(str(default))
    if usable(fallback):
        return Casting(fallback, FROM_DEFAULT,
                       f"nothing in the brief chose a {role}, so the default")

    # Last resort: anything at all that can play the part, said out loud
    # rather than returning None and letting a caller guess.
    from .instruments import all_instruments
    for instrument in all_instruments():
        if usable(instrument):
            return Casting(instrument, FROM_DEFAULT,
                           f"the only available {role}")
    raise LookupError(f"no instrument in the catalogue can play {role!r}")
