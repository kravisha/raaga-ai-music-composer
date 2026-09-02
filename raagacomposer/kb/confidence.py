"""How sure we are, and why - specification sections 10 and 41.

Section 10 lists eight things confidence should consider.  A single number
computed from them and then stored alone cannot answer section 41's question,
"why did you believe that?", so what is stored here is the number *and* the
components it came from.  A person looking at 0.72 can see that it is two
independent sources, directly demonstrated, with one contradiction against it.

Two rules that keep the number honest:

*Confidence never replaces provenance.*  Section 10 says so outright.  Nothing
here lets a high number stand in for knowing where something came from; the
score is computed from the evidence records and is meaningless without them.

*Independence is counted, not source count.*  Ten evidence records from one
video are one source agreeing with itself.  What raises confidence is separate
sources agreeing, and the arithmetic reflects that: supporting evidence has
diminishing returns within a source and real weight across sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .models import Evidence, ExtractionMethod, confidence_band

#: What each consideration can contribute.  They sum to 1.0 so the result is
#: readable as a fraction, and every one of section 10's factors appears.
WEIGHTS = {
    "source_quality": 0.18,
    "independent_sources": 0.22,
    "direct_demonstration": 0.20,
    "agreement": 0.12,
    "extraction_quality": 0.13,
    "human_confirmation": 0.15,
}
#: Subtracted rather than weighted: a contradiction should be able to pull a
#: claim down past what its supporters can push it to.
CONTRADICTION_PENALTY = 0.30
AMBIGUITY_PENALTY = 0.10


@dataclass
class ConfidenceResult:
    value: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def band(self) -> str:
        return confidence_band(self.value)

    def explain(self) -> str:
        """Section 41 - the number, in words."""
        if not self.parts:
            return f"{self.value:.2f} ({self.band}); nothing recorded about why"
        ranked = sorted(self.parts.items(), key=lambda kv: -abs(kv[1]))
        pieces = [f"{name.replace('_', ' ')} {value:+.2f}"
                  for name, value in ranked if abs(value) >= 0.005]
        text = f"{self.value:.2f} ({self.band}) from " + ", ".join(pieces)
        if self.notes:
            text += ". " + " ".join(self.notes)
        return text


def score(evidence: Sequence[Evidence], *,
          source_quality: float = 0.6,
          human_confirmed: bool = False,
          human_rejected: bool = False,
          agrees_with_existing: Optional[bool] = None,
          ambiguous: bool = False) -> ConfidenceResult:
    """Work out a confidence, and keep the working.

    ``agrees_with_existing`` is three-valued on purpose: ``True`` when this
    claim matches something already held, ``False`` when it contradicts one,
    and ``None`` when there was nothing to compare it against - which is not
    the same as disagreement and must not be scored as if it were.
    """
    result = ConfidenceResult()
    supporting = [e for e in evidence if e.supports]
    against = [e for e in evidence if not e.supports]

    if not supporting:
        result.value = 0.0
        result.notes.append("nothing supports this yet")
        return result

    # -- source quality -------------------------------------------------
    result.parts["source_quality"] = round(
        WEIGHTS["source_quality"] * max(0.0, min(1.0, source_quality)), 4)

    # -- independent sources, with diminishing returns ------------------
    sources = {e.source_id for e in supporting if e.source_id}
    independent = max(1, len(sources))
    # 1 source -> 0.45, 2 -> 0.72, 3 -> 0.86, 4+ -> approaching 1.
    independence = 1.0 - 0.55 ** independent
    result.parts["independent_sources"] = round(
        WEIGHTS["independent_sources"] * independence, 4)
    if independent > 1:
        result.notes.append(f"{independent} independent sources agree.")

    # -- demonstrated or reasoned ---------------------------------------
    observed = [e for e in supporting if e.observed]
    demonstrated = len(observed) / len(supporting)
    result.parts["direct_demonstration"] = round(
        WEIGHTS["direct_demonstration"] * demonstrated, 4)
    if not observed:
        result.notes.append("Nothing here was directly observed.")

    # -- agreement with what is already held ----------------------------
    if agrees_with_existing is True:
        agreement = 1.0
    elif agrees_with_existing is False:
        agreement = 0.0
    else:
        agreement = 0.5          # nothing to compare against
    result.parts["agreement"] = round(WEIGHTS["agreement"] * agreement, 4)

    # -- how well it was extracted --------------------------------------
    strength = sum(e.strength for e in supporting) / len(supporting)
    result.parts["extraction_quality"] = round(
        WEIGHTS["extraction_quality"] * max(0.0, min(1.0, strength)), 4)

    # -- a person having looked -----------------------------------------
    if human_rejected:
        result.parts["human_confirmation"] = -WEIGHTS["human_confirmation"]
        result.notes.append("A person marked this incorrect.")
    elif human_confirmed:
        result.parts["human_confirmation"] = WEIGHTS["human_confirmation"]
        result.notes.append("Confirmed by a person.")
    else:
        result.parts["human_confirmation"] = 0.0

    # -- what pulls it down ---------------------------------------------
    if against:
        contradicting_sources = {e.source_id for e in against if e.source_id}
        penalty = CONTRADICTION_PENALTY * min(
            1.0, max(1, len(contradicting_sources)) / independent)
        result.parts["contradiction"] = -round(penalty, 4)
        result.notes.append(
            f"{len(against)} piece(s) of evidence argue against it.")
    if ambiguous:
        result.parts["ambiguity"] = -AMBIGUITY_PENALTY
        result.notes.append("The material was ambiguous.")

    total = sum(result.parts.values())
    result.value = round(max(0.0, min(1.0, total)), 3)
    return result


def blend_for_display(items: Iterable[float]) -> float:
    """The confidence of a set, for a health report or a context block."""
    values = [v for v in items if v is not None]
    return round(sum(values) / len(values), 3) if values else 0.0
