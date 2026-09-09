"""Syllables onto melody events — contract `5ee63d7` §1.3, §6.

Two reproduced defects live in this stage in the reviewed implementation, and
both are invariants here rather than warnings:

* **Nothing extends past the end of the tune.** Four syllables over three
  one-second notes finished at four seconds there. A phrase that runs past its
  melody is not that melody.
* **A rest stays a rest.** A syllable held across a rest filled the silence with
  nominal pitch there. Keeping an unquantised f0 array is not enough if this
  stage hands a later one a span that crosses silence.

The unit of alignment is the **sung event**. Rests are never assigned a
syllable, and no syllable's span crosses one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


class AlignmentError(Exception):
    """A refusal. Never a silent correction."""


@dataclass
class Placed:
    """One syllable on one span of the tune, inside one sung stretch."""
    index: int
    text: str
    onset: str
    vowel: str
    coda: str
    start_s: float
    end_s: float
    midi: float
    #: Which contiguous sung stretch this sits in. Two syllables in different
    #: stretches are separated by a rest, and nothing may bridge them.
    phrase: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def sung_stretches(tune) -> list:
    """Contiguous runs of sung events, split wherever a rest interrupts.

    A gap between events is a rest whether or not anybody wrote one down, so
    both explicit rests and silence between events break a stretch.
    """
    stretches, current = [], []
    previous_end = None
    for event in tune.events:
        if event.get("midi") is None:
            if current:
                stretches.append(current)
                current = []
            previous_end = event["start_s"] + event["duration_s"]
            continue
        if previous_end is not None and event["start_s"] > previous_end + 1e-6:
            if current:
                stretches.append(current)
                current = []
        current.append(event)
        previous_end = event["start_s"] + event["duration_s"]
    if current:
        stretches.append(current)
    return stretches


def place(syllables: list, tune, *, allow_melisma: bool = True) -> list:
    """Place every syllable, or refuse and say why.

    Syllables are distributed across the sung stretches in proportion to how
    many events each holds, so a phrase of two notes does not receive half the
    lyric. Within a stretch, syllables that outnumber events subdivide those
    events; events that outnumber syllables are held (melisma), which is the
    ordinary case for raga.
    """
    if not syllables:
        raise AlignmentError(
            "No syllables to place. An empty lyric is refused rather than "
            "rendered as a successful silence.")
    stretches = sung_stretches(tune)
    if not stretches:
        raise AlignmentError("The tune holds no sung events; there is nothing to sing on.")

    total_events = sum(len(s) for s in stretches)
    if len(syllables) > total_events * 8:
        raise AlignmentError(
            f"{len(syllables)} syllables over {total_events} events would give "
            f"each less than an eighth of a note. Shorten the lyric or lengthen "
            f"the tune; this stage will not compress a lyric until it is unsayable.")

    shares = _shares(len(syllables), [len(s) for s in stretches])
    placed, cursor = [], 0
    for phrase_index, (stretch, take) in enumerate(zip(stretches, shares)):
        if take == 0:
            continue
        chunk = syllables[cursor:cursor + take]
        cursor += take
        placed += _place_in_stretch(chunk, stretch, phrase_index, allow_melisma)

    if cursor != len(syllables):          # defensive; _shares must allocate all
        raise AlignmentError(
            f"{len(syllables) - cursor} syllable(s) were not placed. Refusing "
            f"rather than letting them ride past the end of the tune.")

    end = max(e["start_s"] + e["duration_s"] for s in stretches for e in s)
    for item in placed:
        if item.end_s > end + 1e-6:
            raise AlignmentError(
                f"{item.text!r} would end at {item.end_s:.3f}s, past the tune's "
                f"{end:.3f}s. Nothing sings past the melody.")
    return placed


def _shares(count: int, capacities: list) -> list:
    """Split `count` syllables across stretches by capacity, losing none.

    Largest-remainder, so the allocation sums exactly and a short stretch is not
    starved to zero while syllables remain.
    """
    total = sum(capacities)
    exact = [count * capacity / total for capacity in capacities]
    shares = [int(value) for value in exact]
    remainder = count - sum(shares)
    order = sorted(range(len(capacities)),
                   key=lambda i: (exact[i] - shares[i], capacities[i]), reverse=True)
    for i in order[:remainder]:
        shares[i] += 1
    # Every non-empty stretch that can take a syllable should have one before
    # any other stretch takes a second.
    for i, share in enumerate(shares):
        if share == 0 and sum(shares) < count:
            shares[i] = 1
    return shares


def _place_in_stretch(syllables: list, events: list, phrase: int,
                      allow_melisma: bool) -> list:
    """Inside one contiguous sung stretch, which never contains a rest."""
    placed = []
    if len(syllables) <= len(events):
        per = len(events) / len(syllables)
        for i, syllable in enumerate(syllables):
            first = int(round(i * per))
            last = int(round((i + 1) * per)) - 1
            last = max(first, min(last, len(events) - 1))
            if not allow_melisma:
                last = first
            placed.append(_placed(len(placed), syllable, phrase,
                                  events[first]["start_s"],
                                  events[last]["start_s"] + events[last]["duration_s"],
                                  events[first]["midi"]))
    else:
        # More syllables than events: subdivide events, and give the remainder
        # to the earliest events rather than spilling past the last one.
        counts = _shares(len(syllables), [1] * len(events))
        cursor = 0
        for event, take in zip(events, counts):
            if take <= 0:
                continue
            step = event["duration_s"] / take
            for j in range(take):
                syllable = syllables[cursor]
                cursor += 1
                start = event["start_s"] + j * step
                placed.append(_placed(len(placed), syllable, phrase,
                                      start, start + step, event["midi"]))
    return placed


def _placed(index: int, syllable, phrase: int, start: float, end: float,
            midi: float) -> Placed:
    return Placed(index=index, text=syllable.text, onset=syllable.onset or "",
                  vowel=syllable.vowel, coda=_first_coda(syllable),
                  start_s=round(float(start), 6), end_s=round(float(end), 6),
                  midi=float(midi), phrase=phrase)


def _first_coda(syllable) -> str:
    codas = getattr(syllable, "codas", None)
    if codas:
        return str(codas[0])
    return str(getattr(syllable, "coda", "") or "")


def save(placed: list, path: Path) -> Path:
    """Written as a flat list, so a person can edit it and re-render."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in placed], indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path
