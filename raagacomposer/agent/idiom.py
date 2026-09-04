"""The idiom of a raaga, as heard: how its phrases tend to move.

Section 37 of the specification lists "phrase tendencies, ascent/descent
behavior, cadence behavior" among the things known about a raaga.  Until
now the composer knew a studied raaga only as a list of phrases to quote.
``RaagaIdiom`` is what those phrases have in common: from each note, how
often the next one steps up, steps down, leaps or stays; which notes phrases
come to rest on; how long they run; what shape they take.  The melody
engine consults it in exactly the places it used to flip a coin.

It is computed from the phrase bank, weighted by each phrase's confidence,
and attached only to the learned view of a raaga (``learned_raaga``).  A
raaga the agent has not studied has no idiom, and composes exactly as it
did before.  Everything here is deterministic: the same phrases give the
same idiom, and every choice that needs randomness takes exactly one draw
from the caller's generator.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..raaga.library import Raaga, parse_swara

# Fewer heard phrases than this say nothing about the raaga's habits.
MIN_PHRASES = 3
MOVES = ("up1", "up2", "same", "down1", "down2")
# The composer's own habits, as pseudo-observations: up or down 40% each,
# stay 20%, a leap one time in four.  Heard moves are added to these, so
# an idiom built from a handful of phrases shades the composer's choices
# rather than replacing them, and every move stays possible.
PRIOR_WEIGHT = 4.0
PRIOR = {"up1": 0.30, "up2": 0.10, "same": 0.20, "down1": 0.30, "down2": 0.10}


def _move(delta: int) -> str:
    if delta == 0:
        return "same"
    if delta == 1:
        return "up1"
    if delta == -1:
        return "down1"
    return "up2" if delta > 0 else "down2"


def _contour(degrees: Sequence[int]) -> str:
    if len(degrees) < 3:
        return "flat"
    peak = max(range(len(degrees)), key=lambda i: degrees[i])
    trough = min(range(len(degrees)), key=lambda i: degrees[i])
    rise = degrees[-1] - degrees[0]
    if 0 < peak < len(degrees) - 1 and degrees[peak] > max(degrees[0], degrees[-1]):
        return "arch"
    if 0 < trough < len(degrees) - 1 and degrees[trough] < min(degrees[0], degrees[-1]):
        return "dip"
    if rise > 0:
        return "rise"
    if rise < 0:
        return "fall"
    return "flat"


@dataclass
class RaagaIdiom:
    transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    endings: Dict[str, float] = field(default_factory=dict)
    contours: Dict[str, float] = field(default_factory=dict)
    mean_length: float = 0.0
    phrases: int = 0

    # -- building ----------------------------------------------------------
    @classmethod
    def from_phrases(cls, raaga: Raaga, phrases: Sequence,
                     min_phrases: int = MIN_PHRASES) -> Optional["RaagaIdiom"]:
        """``phrases`` are objects with ``swaras`` and ``confidence`` (the
        repository's ``Phrase``) or plain swara lists.  Returns None when
        there are too few to say anything."""
        rows: List[tuple] = []
        for phrase in phrases:
            swaras = getattr(phrase, "swaras", phrase)
            weight = float(getattr(phrase, "confidence", 1.0))
            if len(swaras) >= 2 and weight > 0:
                rows.append((list(swaras), weight))
        if len(rows) < min_phrases:
            return None

        transitions: Dict[str, Dict[str, float]] = {}
        endings: Dict[str, float] = {}
        contours: Dict[str, float] = {}
        total_length = 0.0
        total_weight = 0.0
        for swaras, weight in rows:
            degrees = [raaga.degree(s) for s in swaras]
            bases = [parse_swara(s)[0] for s in swaras]
            for a, b, da, db in zip(bases, bases[1:], degrees, degrees[1:]):
                row = transitions.setdefault(a, {m: 0.0 for m in MOVES})
                row[_move(db - da)] += weight
            endings[bases[-1]] = endings.get(bases[-1], 0.0) + weight
            shape = _contour(degrees)
            contours[shape] = contours.get(shape, 0.0) + weight
            total_length += len(swaras) * weight
            total_weight += weight

        return cls(
            transitions=transitions,
            endings={k: round(v / total_weight, 4) for k, v in endings.items()},
            contours={k: round(v / total_weight, 4) for k, v in contours.items()},
            mean_length=round(total_length / total_weight, 3),
            phrases=len(rows),
        )

    # -- reading -----------------------------------------------------------
    def move_weights(self, token: str) -> Dict[str, float]:
        """Weights of the five moves from ``token``: what was heard, on top
        of the composer's prior."""
        base = parse_swara(token)[0]
        row = self.transitions.get(base, {})
        return {m: row.get(m, 0.0) + PRIOR[m] * PRIOR_WEIGHT for m in MOVES}

    def pick_direction(self, token: str, rng: random.Random) -> int:
        """+1, -1 or 0, weighted by what followed this note in the phrases
        heard.  Exactly one draw."""
        w = self.move_weights(token)
        up, down, same = w["up1"] + w["up2"], w["down1"] + w["down2"], w["same"]
        r = rng.random() * (up + down + same)
        if r < up:
            return 1
        if r < up + down:
            return -1
        return 0

    def pick_steps(self, token: str, direction: int, rng: random.Random) -> int:
        """1 or 2 scale degrees in ``direction``, weighted the same way.
        Exactly one draw."""
        w = self.move_weights(token)
        one, two = (w["up1"], w["up2"]) if direction > 0 else (w["down1"], w["down2"])
        return 1 if rng.random() * (one + two) < one else 2

    def cadence_for(self, raaga: Raaga, token: str) -> str:
        """The resting note to close on from ``token``: the nyasa the heard
        phrases most often end on, discounted by distance.  Deterministic."""
        if not raaga.nyasa:
            return token
        base, octave = parse_swara(token)
        deg = raaga.degree(token)
        best, best_score = token, -1.0
        for nyasa in raaga.nyasa:
            share = self.endings.get(nyasa, 0.0)
            for o in (octave - 1, octave, octave + 1):
                cand = nyasa + ("+" * o if o > 0 else "-" * -o)
                distance = abs(raaga.degree(cand) - deg)
                score = (share + 0.05) / (1.0 + distance)
                if score > best_score:
                    best_score, best = score, cand
        return best

    def describe(self) -> str:
        if not self.phrases:
            return "no idiom"
        rests = ", ".join(f"{k} {v:.0%}" for k, v in
                          sorted(self.endings.items(), key=lambda kv: -kv[1])[:3])
        shapes = ", ".join(f"{k} {v:.0%}" for k, v in
                           sorted(self.contours.items(), key=lambda kv: -kv[1])[:3])
        return (f"{self.phrases} phrases heard, about {self.mean_length:.1f} "
                f"notes each; rests on {rests}; shapes {shapes}")
