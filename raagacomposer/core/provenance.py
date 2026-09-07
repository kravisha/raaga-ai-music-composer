"""Where a piece of knowledge came from, and what may be done with it.

Training architecture specification sections 2.2 to 2.5, and non-negotiable
rules 1 and 2: *training data and generated output are distinct*, and
generated output is reusable for composition but is not automatic
same-agent retraining input.

The distinction had no representation in the store.  A phrase carried a
``source_id`` and a free-text ``notes``, so material the agent had written
itself sat in the same pool as material learned from a recording, and the
only thing telling them apart was English prose nothing reads.  That is
not a rule, it is a hope; this module makes it a field.

The three origins are deliberately about *who produced the material*, not
about how good it is or whether it was approved.  Quality is ``confidence``
and approval is ``rejected``; conflating them with origin is what lets a
well-rated generation quietly become evidence.
"""
from __future__ import annotations

from typing import Tuple

#: A person produced it: a recording, a transcription, the curated library,
#: or the creator saying so directly.
HUMAN = "human"

#: Found online.  Still made by people, but its trustworthiness has not been
#: established, so specification section 16 keeps it a *candidate* until it
#: is judged.  Kept separate from HUMAN so that judgement stays possible.
INTERNET = "internet"

#: The application's own rendering of the grammar it ships with - the
#: reference library played aloud so the agent can practise hearing it.
#: Trainable, because the *content* is human-authored musicology and the
#: agent is doing ear training on known-correct material, the way a student
#: plays scales from a book.  Kept apart from HUMAN because nobody
#: performed it: counting it as a person's recording would overstate what
#: the agent has actually heard.
REFERENCE = "reference"

#: Nobody recorded where this came from.  Distinct from GENERATED, which
#: is a claim ("we made it"); this is the absence of a claim, and it is
#: not trainable for the same reason an unrecognised origin is not: a
#: value nobody established must never widen what the agent learns from.
UNKNOWN = "unknown"

#: This system produced it.  Legitimate creative material (section 2.3) and
#: legitimate to compose from, reuse, index for originality and show to the
#: creator - but never evidence that the agent has learned anything, because
#: the only thing it evidences is that the agent repeated itself.
GENERATED = "generated"

ORIGINS: Tuple[str, ...] = (HUMAN, INTERNET, REFERENCE, GENERATED,
                            UNKNOWN)

#: What may be treated as something the agent *learned from*.  The whole
#: point of the module is that this tuple does not contain GENERATED.
#: REFERENCE is in it: shipped grammar rendered aloud is teaching material,
#: not invention.  What is excluded is only what the agent made up itself.
LEARNED_FROM: Tuple[str, ...] = (HUMAN, INTERNET, REFERENCE)

_DESCRIPTIONS = {
    HUMAN: "learned from a person's recording",
    INTERNET: "found online, pending judgement",
    REFERENCE: "the shipped library, rendered for practice",
    UNKNOWN: "of unrecorded origin",
    GENERATED: "written by this system",
}


def is_valid(origin: str) -> bool:
    return origin in ORIGINS


def may_be_learned_from(origin: str) -> bool:
    """Whether material of this origin can count as evidence of learning.

    An unknown origin is *not* trainable.  A value nobody recognises is a
    value nobody checked, and the conservative reading is the one that
    cannot silently widen what the agent trains on (specification 20).
    """
    return origin in LEARNED_FROM


def describe(origin: str) -> str:
    return _DESCRIPTIONS.get(origin, f"of unrecorded origin ({origin!r})")


#: What a source *is*, as against what has been learned from it.  Listing a
#: queued or failed recording as "learned from a person's recording" states
#: the outcome of an analysis that has not happened, under a heading saying
#: it has not happened.  These describe the thing on file; ``describe``
#: describes the standing of what came out of it.
_SOURCE_KINDS = {
    HUMAN: "a person's recording",
    INTERNET: "material found online",
    REFERENCE: "the shipped library's reference material",
    UNKNOWN: "of unrecorded origin",
    GENERATED: "written by this system",
}


def describe_source(origin: str) -> str:
    """What kind of thing a source is, making no claim about learning."""
    return _SOURCE_KINDS.get(origin, f"of unrecorded origin ({origin!r})")


def coerce(origin: str, *, source_id: str = "") -> str:
    """Settle a phrase's origin, preferring what can be derived over what
    was merely left at its default.

    A phrase learned from a source always carries that source's id.  So a
    phrase claiming to be learned while naming nothing it was learned from
    has not been labelled, it has been *defaulted* - and the dangerous
    direction of that mistake is only one: generated material counted as
    human evidence.  Deriving it here cannot mislabel a genuinely learned
    phrase, because a genuinely learned phrase has a source.
    """
    origin = (origin or "").strip().lower()
    if not is_valid(origin):
        return GENERATED if not source_id else HUMAN
    # Every trainable origin, REFERENCE included, has to name what it came
    # from.  An earlier version of this returned REFERENCE before reaching
    # this check, which let a source-less reference phrase into the learned
    # pool without the evidence the rule exists to require.
    if origin in LEARNED_FROM and not source_id:
        return GENERATED
    return origin
