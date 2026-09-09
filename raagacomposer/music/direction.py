"""A direction for one rewrite: the musical property asked for, as controls.

"Make the Charanam softer and plainer", "give the Pallavi a clearer
cadence in a closer register": a section is named (music/structure and
production/targets find it) and a property is asked of it.  This reads
the property out of the words and turns it into the handful of controls
the generator actually has - a register window, a step preference, a
resting cadence, how much ornament, how much energy, whether to vary.  The
mapping is small, deterministic and disclosed by ``describe``; a word that
names no control is reported as such, never guessed at.  An emotion word
("sadder", "tender") maps to controls - softer, closer, resting - and the
reply says those controls, not that sadness was understood or achieved.

A direction is for one rewrite and one section.  It is laid over the
lesson guidance for that rewrite only: explicit here beats a soft
preference there (less ornament beats a lesson's add_gamaka), and a hard
restriction there (a swara or transition to avoid, an ending to avoid)
is never lifted here; a request that cannot coexist with one is reported
as a conflict.  Nothing stored - lessons, the brief, the stored options -
is changed.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

#: Word -> (control, value).  Longest phrase first when reading, so
#: "fewer leaps" is not read as "leaps".
_VOCABULARY: List[Tuple[str, str, str]] = [
    ("closer register", "register", "closer"),
    ("narrower register", "register", "closer"),
    ("closer", "register", "closer"),
    ("narrower", "register", "closer"),
    ("lower register", "register", "lower"),
    ("higher register", "register", "higher"),
    ("lower", "register", "lower"),
    ("higher", "register", "higher"),
    ("fewer leaps", "motion", "stepwise"),
    ("no leaps", "motion", "stepwise"),
    ("stepwise", "motion", "stepwise"),
    ("by step", "motion", "stepwise"),
    ("smoother contour", "motion", "stepwise"),
    ("smoother", "motion", "stepwise"),
    ("more leaps", "motion", "leaping"),
    ("leaping", "motion", "leaping"),
    ("clearer cadence", "cadence", "resting"),
    ("clear cadence", "cadence", "resting"),
    ("resting cadence", "cadence", "resting"),
    ("land on the sa", "cadence", "resting"),
    ("rest at the end", "cadence", "resting"),
    ("come to rest", "cadence", "resting"),
    ("resolve", "cadence", "resting"),
    ("more gamaka", "ornament", "more"),
    ("more ornament", "ornament", "more"),
    ("more ornamented", "ornament", "more"),
    ("more ornamentation", "ornament", "more"),
    ("less gamaka", "ornament", "less"),
    ("less ornament", "ornament", "less"),
    ("less ornamented", "ornament", "less"),
    ("less ornamentation", "ornament", "less"),
    ("plainer", "ornament", "less"),
    ("plain", "ornament", "less"),
    ("simpler", "ornament", "less"),
    ("softer", "energy", "softer"),
    ("quieter", "energy", "softer"),
    ("gentler", "energy", "softer"),
    ("more inward", "energy", "softer"),
    ("stronger", "energy", "stronger"),
    ("louder", "energy", "stronger"),
    ("more energetic", "energy", "stronger"),
    ("more energy", "energy", "stronger"),
    ("brighter", "energy", "stronger"),
    ("more variation", "variety", "more"),
    ("more varied", "variety", "more"),
    ("less repetitive", "variety", "more"),
    ("vary it more", "variety", "more"),
]

#: Emotion words: each is a bundle of controls, and the reply names the
#: controls.  Nothing here claims to know what sad is.
_EMOTIONS = {
    "sadder": (("energy", "softer"), ("register", "closer"), ("cadence", "resting")),
    "sad": (("energy", "softer"), ("register", "closer"), ("cadence", "resting")),
    "tender": (("energy", "softer"), ("register", "closer")),
    "more tender": (("energy", "softer"), ("register", "closer")),
    "calmer": (("energy", "softer"), ("motion", "stepwise")),
    "happier": (("energy", "stronger"), ("register", "higher")),
    "more joyful": (("energy", "stronger"), ("register", "higher")),
    "more dramatic": (("energy", "stronger"), ("motion", "leaping")),
}

#: Words that ask for a musical property this engine has no control for.
#: Reported by name so a Critic or a creator knows what was not done.
_UNSUPPORTED = re.compile(
    r"\b(?:faster|slower|tempo|rubato|syncopat\w*|swing|harmon\w*|chord\w*|"
    r"modulat\w*|key change|breath\w*|vibrato|timbre|tone colou?r|"
    r"louder chorus|call and response|counterpoint|polyphon\w*|"
    r"more silence|more space|longer notes|shorter notes|"
    r"different raaga|another raaga|change the raaga)\b")


@dataclass
class SectionDirection:
    register: str = ""      # lower | higher | closer
    motion: str = ""        # stepwise | leaping
    cadence: str = ""       # resting
    ornament: str = ""      # more | less
    energy: str = ""        # softer | stronger
    variety: str = ""       # more
    words: List[str] = field(default_factory=list)        # what set the controls
    unsupported: List[str] = field(default_factory=list)  # property words with no control

    def is_empty(self) -> bool:
        return not (self.register or self.motion or self.cadence or self.ornament
                    or self.energy or self.variety)

    def controls(self) -> List[str]:
        out = []
        if self.energy:
            out.append(f"{self.energy} (velocity and density)")
        if self.register:
            out.append({"closer": "closer register (window narrowed)",
                        "lower": "lower register (window moved down)",
                        "higher": "higher register (window moved up)"}[self.register])
        if self.motion:
            out.append("stepwise (fewer leaps)" if self.motion == "stepwise"
                       else "leaping (the walk left free)")
        if self.cadence:
            out.append("resting cadence (ends on a resting note)")
        if self.ornament:
            out.append("more gamaka" if self.ornament == "more" else "less gamaka")
        if self.variety:
            out.append("more variety")
        return out

    def describe(self) -> str:
        """The controls pulled, and the words that were not controls."""
        parts = []
        if not self.is_empty():
            parts.append("; ".join(self.controls()))
        if self.unsupported:
            parts.append("not a control I have: " + ", ".join(self.unsupported))
        return ". ".join(parts)


def read_direction(text: str) -> SectionDirection:
    """The controls a sentence asks for.  Later words win over earlier ones
    for the same control ("softer, no, stronger" is stronger)."""
    direction = SectionDirection()
    lowered = (text or "").lower()
    covered = lowered
    hits: List[Tuple[int, str, str, str]] = []
    for phrase, control, value in sorted(_VOCABULARY, key=lambda v: -len(v[0])):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", covered):
            hits.append((match.start(), control, value, phrase))
            covered = covered[:match.start()] + " " * len(phrase) + covered[match.end():]
    for word, bundle in sorted(_EMOTIONS.items(), key=lambda kv: -len(kv[0])):
        for match in re.finditer(rf"\b{re.escape(word)}\b", covered):
            for control, value in bundle:
                hits.append((match.start(), control, value, word))
            covered = covered[:match.start()] + " " * len(word) + covered[match.end():]
    for _, control, value, phrase in sorted(hits, key=lambda h: h[0]):
        setattr(direction, control, value)
        if phrase not in direction.words:
            direction.words.append(phrase)
    for match in _UNSUPPORTED.finditer(lowered):
        word = match.group(0)
        if word not in direction.unsupported:
            direction.unsupported.append(word)
    return direction


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------
def directed_register(lo: int, hi: int, low: int, high: int,
                      register: str) -> Tuple[int, int, str]:
    """The section's window under a register direction, kept inside the
    usable voice window ``low``-``high`` and never narrower than a fifth.
    Returns (lo, hi, conflict) - the conflict text when the direction
    could not be honoured."""
    if not register:
        return lo, hi, ""
    if register == "closer":
        width = hi - lo
        if width <= 7:
            return lo, hi, "the register is already within a fifth"
        trim = max(1, int(round(width * 0.2)))
        return lo + trim, hi - trim, ""
    shift = -5 if register == "lower" else 5
    new_lo, new_hi = lo + shift, hi + shift
    if register == "lower" and new_lo < low:
        new_lo, new_hi = low, max(low + 7, hi - (lo - low))
        if new_lo == lo:
            return lo, hi, "the singer's range has nothing below this section's window"
    if register == "higher" and new_hi > high:
        new_hi, new_lo = high, min(high - 7, lo + (high - hi))
        if new_hi == hi:
            return lo, hi, "the singer's range has nothing above this section's window"
    return new_lo, new_hi, ""


@dataclass
class DirectedRewrite:
    opts: Any
    guidance: Any
    conflicts: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)


def apply_direction(direction: SectionDirection, opts: Any, guidance: Any,
                    raaga: Any = None) -> DirectedRewrite:
    """Options and guidance for one rewrite under ``direction``: copies,
    never the originals.  Explicit here beats a soft preference in the
    lesson guidance; a hard restriction there stays and a request that
    cannot live with it is reported."""
    new_opts = copy.copy(opts)
    new_guidance = copy.deepcopy(guidance) if guidance is not None else None
    result = DirectedRewrite(opts=new_opts, guidance=new_guidance)
    if direction is None or direction.is_empty():
        return result
    result.controls = direction.controls()
    new_opts.direction = direction
    if direction.motion == "stepwise" and new_guidance is not None:
        new_guidance.prefer_step = max(float(getattr(new_guidance, "prefer_step", 0.0)), 0.9)
    if direction.variety == "more" and new_guidance is not None:
        new_guidance.vary_more = True
    if direction.ornament == "less" and new_guidance is not None:
        # An explicit "plainer" beats a lesson's soft add_gamaka.
        new_guidance.add_gamaka = False
    if direction.ornament == "more" and new_guidance is not None:
        new_guidance.add_gamaka = True
    if direction.cadence == "resting" and new_guidance is not None:
        nyasa = list(getattr(raaga, "nyasa", None) or []) if raaga is not None else []
        avoided = set(getattr(new_guidance, "avoid_endings", None) or ())
        if nyasa and all(n in avoided for n in nyasa):
            result.conflicts.append(
                "a resting cadence: every resting note of the raaga is one a "
                "lesson says not to end on, so the lesson stands")
        else:
            new_guidance.must_end_on_nyasa = True
    if direction.motion == "stepwise" and new_guidance is not None:
        transitions = set(getattr(new_guidance, "avoid_transitions", None) or ())
        if transitions:
            result.controls.append(
                f"stepwise within the lessons' {len(transitions)} forbidden move(s)")
    return result
