"""The raaga as the agent actually knows it, rebuilt from memory.

Everything downstream - practice, evaluation and composition - works from this
view rather than from the shipped library, so what the agent has learned is
what it plays.  Where memory is silent the shipped definition fills the gap,
and the caller is told how much of the picture came from learning.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..raaga.library import Raaga, RaagaLibrary, parse_swara
from .idiom import RaagaIdiom
from .knowledge import KnowledgeRepository

log = get_logger("agent.learned")

# Facts that make up a usable working picture of a raaga.
CORE_KEYS = ("arohanam", "avarohanam", "swaras", "jeeva", "nyasa", "gamaka")


def _tokens(value: str) -> List[str]:
    return [t for t in value.replace(",", " ").split() if t]


def learned_raaga(repo: KnowledgeRepository, library: RaagaLibrary,
                  name: str, min_confidence: float = 0.4
                  ) -> Tuple[Optional[Raaga], float]:
    """Assemble a Raaga from stored facts and learned phrases.

    Returns (raaga, completeness) where completeness is the share of the core
    facts that came from the repository rather than from the fallback.
    """
    fallback = library.get(name)
    facts = {f.key: f for f in repo.facts(name)
             if f.confidence >= min_confidence}
    if not facts and fallback is None:
        return None, 0.0
    if fallback is None and "arohanam" not in facts:
        return None, 0.0

    base = fallback or Raaga(name=name)
    known = 0

    arohanam = _tokens(facts["arohanam"].value) if "arohanam" in facts else \
        list(base.arohanam)
    known += int("arohanam" in facts)
    avarohanam = _tokens(facts["avarohanam"].value) if "avarohanam" in facts else \
        list(base.avarohanam)
    known += int("avarohanam" in facts)
    jeeva = _tokens(facts["jeeva"].value) if "jeeva" in facts else list(base.jeeva)
    known += int("jeeva" in facts)
    nyasa = _tokens(facts["nyasa"].value) if "nyasa" in facts else list(base.nyasa)
    known += int("nyasa" in facts)
    graha = _tokens(facts["graha"].value) if "graha" in facts else list(base.graha)
    known += int("swaras" in facts)

    gamaka: Dict[str, str] = dict(base.gamaka)
    if "gamaka" in facts:
        parsed: Dict[str, str] = {}
        for item in facts["gamaka"].value.split(","):
            swara, _, kind = item.strip().partition(":")
            if swara and kind:
                parsed[swara.strip()] = kind.strip()
        if parsed:
            gamaka = parsed
            known += 1

    moods = [m.strip().lower() for m in facts["moods"].value.split(",")] \
        if "moods" in facts else list(base.moods)

    # Characteristic phrases: what the agent has actually heard, ranked by
    # confidence, ahead of anything the library merely asserts.
    # A monotone run through the scale is the arohanam or avarohanam being
    # played: a fact the view already carries, not a phrase to quote or to
    # learn habits from.  It stays in the bank the evaluator and the
    # originality checker read; it does not become a prayoga or an idiom.
    heard_phrases = [p for p in
                     repo.phrases(raaga=name, min_confidence=min_confidence, limit=64)
                     if 2 <= len(p.swaras) <= 8 and not _is_scale_run(base, p.swaras)]
    prayogas: List[List[str]] = [list(p.swaras) for p in heard_phrases]
    heard = len(prayogas)
    for phrase in base.prayogas:
        if list(phrase) not in prayogas:
            prayogas.append(list(phrase))

    raaga = replace(
        base,
        name=base.name or name,
        arohanam=arohanam,
        avarohanam=avarohanam,
        jeeva=jeeva,
        nyasa=nyasa,
        graha=graha or base.graha,
        gamaka=gamaka,
        prayogas=prayogas,
        moods=moods,
        source="learned" if known else base.source,
    )
    # Built from the final arohanam/avarohanam/nyasa (facts if known, the
    # library otherwise) so degree() and cadence_for() agree with what the
    # rest of this raaga view says, not with the fallback's ladder.
    idiom = RaagaIdiom.from_phrases(raaga, heard_phrases)
    raaga = replace(raaga, idiom=idiom)
    completeness = round(known / len(CORE_KEYS), 3)
    log.debug("learned view of %s: %.0f%% from memory, %d heard phrases%s",
              name, completeness * 100, heard,
              f"; idiom: {idiom.describe()}" if idiom else "")
    return raaga, completeness


# A monotone run of this many notes or more is the scale itself.
SCALE_RUN_MIN = 6


def _is_scale_run(raaga: Raaga, swaras: Sequence[str]) -> bool:
    if len(swaras) < SCALE_RUN_MIN:
        return False
    degrees = [raaga.degree(s) for s in swaras]
    steps = [b - a for a, b in zip(degrees, degrees[1:])]
    return all(s == 1 for s in steps) or all(s == -1 for s in steps)


def learned_phrase_bank(repo: KnowledgeRepository, name: str,
                        min_confidence: float = 0.4,
                        limit: int = 64) -> List[List[str]]:
    """Phrases the agent has heard, best first."""
    return [list(p.swaras) for p in
            repo.phrases(raaga=name, min_confidence=min_confidence, limit=limit)
            if 2 <= len(p.swaras) <= 10]


def knowledge_confidence(repo: KnowledgeRepository, name: str) -> Dict[str, float]:
    """How well the agent thinks it knows this raaga, by facet."""
    facts = {f.key: f.confidence for f in repo.facts(name)}
    phrases = repo.phrases(raaga=name, limit=500)
    phrase_confidence = (sum(p.confidence for p in phrases) / len(phrases)
                         if phrases else 0.0)
    core = sum(1 for k in CORE_KEYS if k in facts) / len(CORE_KEYS)
    return {
        "core_facts": round(core, 3),
        "fact_confidence": round(
            sum(facts.values()) / len(facts) if facts else 0.0, 3),
        "phrases": len(phrases),
        "phrase_confidence": round(phrase_confidence, 3),
        "overall": round(0.45 * core + 0.25 * min(1.0, len(phrases) / 12.0)
                         + 0.3 * phrase_confidence, 3),
    }


def describe_knowledge(repo: KnowledgeRepository, name: str) -> str:
    """A plain-language account of what the agent knows and where it came from."""
    facts = repo.facts(name)
    if not facts:
        return f"I have not learned anything about {name} yet."
    rows = [f"What I know about {name}:"]
    seen = set()
    for fact in facts:
        if fact.key in seen:
            continue
        seen.add(fact.key)
        source = repo.source(fact.source_id)
        origin = source.title if source else "unknown source"
        flag = "  [disputed]" if fact.disputed else ""
        rows.append(f"  {fact.key:<22} {fact.value}")
        rows.append(f"  {'':<22} from {origin}, confidence "
                    f"{fact.confidence:.2f}{flag}")
    phrases = repo.phrases(raaga=name, limit=8)
    if phrases:
        rows.append(f"  characteristic phrases I have heard ({len(phrases)} shown):")
        for phrase in phrases:
            source = repo.source(phrase.source_id)
            rows.append(f"    {' '.join(phrase.swaras):<28} "
                        f"confidence {phrase.confidence:.2f} "
                        f"({source.title if source else 'unknown'})")
    return "\n".join(rows)
