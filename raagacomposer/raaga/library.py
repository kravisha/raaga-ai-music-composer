"""Raaga knowledge store (spec section 10).

Raaga knowledge is structural data, not something a language model is asked to
remember.  The library ships a curated set and merges any user-supplied file at
``<config>/raagas_user.json`` with the same shape, so the library expands
without touching application code.

Swaras are written as ``S R1 G3 M1 P D1 N3``.  A trailing ``+`` raises by an
octave and ``-`` lowers, so ``S+`` is the upper tonic and ``P-`` the lower
fifth.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from ..kb.normalize import normalise_name

log = get_logger("raaga.library")

DATA_FILE = Path(__file__).with_name("data") / "raagas.json"
#: The 72 parent scales, generated from the Stage 1 knowledge pack by
#: ``tools/build_melakartas.py``.  Grammar the application can rely on;
#: everything curated about a raaga still lives in ``raagas.json``.
MELAKARTA_FILE = Path(__file__).with_name("data") / "melakartas.json"
USER_FILE_NAME = "raagas_user.json"

SWARA_SEMITONES: Dict[str, int] = {}


def parse_swara(token: str) -> Tuple[str, int]:
    """``"S+"`` -> ``("S", 1)``."""
    token = token.strip()
    octave = token.count("+") - token.count("-")
    base = token.replace("+", "").replace("-", "")
    return base, octave


def swara_semitone(token: str) -> int:
    base, octave = parse_swara(token)
    return SWARA_SEMITONES.get(base, 0) + 12 * octave


def swara_midi(token: str, tonic_midi: int, octave_offset: int = 0) -> int:
    return tonic_midi + swara_semitone(token) + 12 * octave_offset


@dataclass
class Raaga:
    name: str
    arohanam: List[str] = field(default_factory=list)
    avarohanam: List[str] = field(default_factory=list)
    jeeva: List[str] = field(default_factory=list)
    nyasa: List[str] = field(default_factory=list)
    graha: List[str] = field(default_factory=list)
    prayogas: List[List[str]] = field(default_factory=list)
    gamaka: Dict[str, str] = field(default_factory=dict)
    avoid: List[List[str]] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    tempo_range: List[int] = field(default_factory=lambda: [60, 100])
    time_of_day: str = "any"
    aliases: List[str] = field(default_factory=list)
    melakarta: Optional[int] = None
    notes: str = ""
    source: str = "builtin"
    # -- the Stage 1 melakarta pack (docs/spec/stage1_knowledge_pack/) -----
    # Grammar: which chakra the melakarta belongs to and the three blocks its
    # scale decomposes into (R-G, madhyama, D-N).
    chakra: str = ""
    rg: str = ""
    madhyama: str = ""
    dn: str = ""
    # Heuristic: what each block is said to colour the raaga with, the pack's
    # starter tags and the uses it suggests.  Kept apart from ``moods``, which
    # is curated, so a learned selection weight can move one and not the other
    # (pack document 05 section 6).
    block_character: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    good_for: List[str] = field(default_factory=list)
    # The learned tendencies (agent/idiom.py, RaagaIdiom); only the learned
    # view (agent/learned.py, learned_raaga) ever carries one.  Excluded from
    # repr/equality and never serialised so it cannot leak into raagas.json.
    idiom: Optional[Any] = field(default=None, repr=False, compare=False)
    # Where each learned prayoga came from (agent/learned.py, learned_raaga),
    # keyed by the phrase's swaras joined with a space: {"phrase_id": ...,
    # "origin": ...}.  Only the learned view ever fills this; excluded from
    # repr/equality and never serialised for the same reason as ``idiom``.
    prayoga_sources: Dict[str, Any] = field(default_factory=dict, repr=False,
                                            compare=False)

    # -- note sets ---------------------------------------------------------
    @property
    def ascending(self) -> List[str]:
        return [parse_swara(s)[0] for s in self.arohanam]

    @property
    def descending(self) -> List[str]:
        return [parse_swara(s)[0] for s in self.avarohanam]

    @property
    def allowed(self) -> List[str]:
        seen: List[str] = []
        for s in self.ascending + self.descending:
            if s not in seen:
                seen.append(s)
        return seen

    @property
    def forbidden_swaras(self) -> List[str]:
        return [a[0] for a in self.avoid if len(a) == 1]

    def is_allowed(self, swara: str) -> bool:
        return parse_swara(swara)[0] in self.allowed

    def allows_ascending(self, swara: str) -> bool:
        return parse_swara(swara)[0] in self.ascending

    def allows_descending(self, swara: str) -> bool:
        return parse_swara(swara)[0] in self.descending

    # -- movement ----------------------------------------------------------
    def _ladder(self, direction: int) -> List[str]:
        seq = self.ascending if direction >= 0 else list(reversed(self.descending))
        out: List[str] = []
        for s in seq:
            if s not in out:
                out.append(s)
        return sorted(out, key=lambda s: SWARA_SEMITONES.get(s, 0))

    def step(self, token: str, steps: int, direction: Optional[int] = None) -> str:
        """Move ``steps`` scale degrees from ``token`` along the raaga ladder."""
        direction = direction if direction is not None else (1 if steps >= 0 else -1)
        ladder = self._ladder(direction)
        base, octave = parse_swara(token)
        if base not in ladder:
            base = min(ladder, key=lambda s: abs(SWARA_SEMITONES.get(s, 0)
                                                 - SWARA_SEMITONES.get(base, 0)))
        idx = ladder.index(base) + steps
        octave += idx // len(ladder)
        base = ladder[idx % len(ladder)]
        return base + ("+" * octave if octave > 0 else "-" * -octave)

    def neighbours(self, token: str, direction: int) -> str:
        return self.step(token, 1 if direction >= 0 else -1, direction)

    def degree(self, token: str) -> int:
        base, octave = parse_swara(token)
        ladder = self._ladder(1)
        idx = ladder.index(base) if base in ladder else 0
        return idx + len(ladder) * octave

    def from_degree(self, degree: int) -> str:
        ladder = self._ladder(1)
        octave, idx = divmod(degree, len(ladder))
        return ladder[idx] + ("+" * octave if octave > 0 else "-" * -octave)

    # -- pitches -----------------------------------------------------------
    def midi(self, token: str, tonic_midi: int) -> int:
        return swara_midi(token, tonic_midi)

    def pitches_in_range(self, tonic_midi: int, low: int, high: int) -> List[int]:
        out: List[int] = []
        for octave in range(-3, 4):
            for s in self.allowed:
                m = tonic_midi + SWARA_SEMITONES.get(s, 0) + 12 * octave
                if low <= m <= high:
                    out.append(m)
        return sorted(set(out))

    def nearest_token(self, midi_note: int, tonic_midi: int) -> str:
        best, best_d = "S", 999
        for octave in range(-3, 4):
            for s in self.allowed:
                m = tonic_midi + SWARA_SEMITONES.get(s, 0) + 12 * octave
                d = abs(m - midi_note)
                if d < best_d:
                    best_d = d
                    best = s + ("+" * octave if octave > 0 else "-" * -octave)
        return best

    # -- what is actually known --------------------------------------------
    @property
    def scale_only(self) -> bool:
        """True when all this raaga has is its scale.

        A melakarta the Stage 1 pack supplies and nobody curated has an
        arohanam, an avarohanam and a block character, and nothing else: no
        jeeva swaras, no resting notes, no prayogas, no gamaka.  Specification
        section 37 says unknown fields stay unknown, so the application says
        so rather than composing as though it knew more.
        """
        return not (self.prayogas or self.jeeva or self.nyasa or self.gamaka)

    def block_summary(self) -> str:
        """The pack's explainable character: block by block, never by name.

        ``"R2G2 tender, introspective, humane; M1 grounded, earthy, settled;
        D1N3 poignant contrast, dramatic upward pull, strong resolution"``.
        """
        parts = [f"{block} {self.block_character[block]}"
                 for block in (self.rg, self.madhyama, self.dn)
                 if block and block in self.block_character]
        return "; ".join(parts)

    def character(self) -> str:
        """One sentence about the raaga, from whatever is actually known.

        A curated note where there is one; otherwise the pack's block logic,
        which is the point of the block model - a reason traceable to the map
        rather than to a name (pack document 01 section F).
        """
        if self.notes:
            return self.notes
        summary = self.block_summary()
        if not summary:
            return ""
        sentence = f"Melakarta {self.melakarta}: {summary}."
        if self.scale_only:
            sentence += (" That is the parent scale and its character; no "
                         "characteristic phrases have been curated or heard "
                         "for it yet.")
        return sentence

    # -- descriptive -------------------------------------------------------
    def gamaka_for(self, token: str) -> str:
        return self.gamaka.get(parse_swara(token)[0], "")

    def mood_score(self, words: Iterable[str]) -> float:
        words = [w.lower() for w in words if w]
        if not words:
            return 0.0
        hits = 0.0
        for m in self.moods:
            for w in words:
                if m in w or w in m:
                    hits += 1.0
                    break
        return hits / max(1.0, len(self.moods) ** 0.5)

    def describe(self) -> str:
        tempo = (f"{self.tempo_range[0]}-{self.tempo_range[-1]} bpm"
                 if self.tempo_range else "not known")
        lines = [self.name,
                 f"  Arohanam:   {' '.join(self.arohanam)}",
                 f"  Avarohanam: {' '.join(self.avarohanam)}",
                 f"  Jeeva swaras: {', '.join(self.jeeva) or '-'}",
                 f"  Resting (nyasa): {', '.join(self.nyasa) or '-'}",
                 f"  Moods: {', '.join(self.moods) or '-'}",
                 f"  Tempo: {tempo}"]
        if self.melakarta:
            chakra = f", chakra {self.chakra}" if self.chakra else ""
            lines.append(f"  Melakarta {self.melakarta}{chakra}")
        summary = self.block_summary()
        if summary:
            lines.append(f"  Character: {summary}")
        if self.scale_only:
            lines.append("  This is the parent scale and its character only - "
                         "no characteristic phrases, resting notes or gamaka "
                         "have been learned or curated for it yet.")
        if self.notes:
            lines.append(f"  {self.notes}")
        return "\n".join(lines)


class RaagaLibrary:
    def __init__(self, extra_path: Optional[Path] = None,
                 melakartas: bool = True) -> None:
        self._raagas: Dict[str, Raaga] = {}
        self._alias: Dict[str, str] = {}
        self.load(DATA_FILE, "builtin")
        if melakartas:
            self.load_melakartas(MELAKARTA_FILE)
        user = Path(extra_path) if extra_path else config_dir() / USER_FILE_NAME
        if user.exists():
            self.load(user, "user")

    # -- loading -----------------------------------------------------------
    def load(self, path: Path, source: str) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("cannot read raaga file %s: %s", path, exc)
            return
        SWARA_SEMITONES.update(data.get("swara_semitones", {}))
        for entry in data.get("raagas", []):
            try:
                raaga = Raaga(
                    name=entry["name"],
                    arohanam=list(entry.get("arohanam", [])),
                    avarohanam=list(entry.get("avarohanam", [])),
                    jeeva=list(entry.get("jeeva", [])),
                    nyasa=list(entry.get("nyasa", [])),
                    graha=list(entry.get("graha", [])),
                    prayogas=[list(p) for p in entry.get("prayogas", [])],
                    gamaka=dict(entry.get("gamaka", {})),
                    avoid=[list(a) for a in entry.get("avoid", [])],
                    moods=[m.lower() for m in entry.get("moods", [])],
                    tempo_range=list(entry.get("tempo_range", [60, 100])),
                    time_of_day=entry.get("time_of_day", "any"),
                    aliases=list(entry.get("aliases", [])),
                    melakarta=entry.get("melakarta"),
                    notes=entry.get("notes", ""),
                    source=source,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("bad raaga entry %r: %s", entry.get("name"), exc)
                continue
            self._raagas[raaga.name.lower()] = raaga
            self._alias[raaga.name.lower()] = raaga.name.lower()
            for a in raaga.aliases:
                self._alias[a.lower()] = raaga.name.lower()
        log.info("raaga library: %d raagas after loading %s", len(self._raagas), path.name)

    def _index(self, raaga: Raaga, *names: str) -> None:
        """Point every spelling of ``raaga`` at its entry, without silence.

        Two raagas claiming one spelling is a real ambiguity - it is how
        "Bhairavi" would quietly become "Natabhairavi" - so a collision is
        logged and the first claim keeps the name.
        """
        key = raaga.name.lower()
        for name in names:
            alias = (name or "").strip().lower()
            if not alias:
                continue
            owner = self._alias.get(alias)
            if owner and owner != key:
                log.warning("raaga alias %r already means %r; leaving it alone "
                            "rather than pointing it at %r", alias, owner, raaga.name)
                continue
            self._alias[alias] = key

    def load_melakartas(self, path: Path = MELAKARTA_FILE) -> None:
        """Merge the Stage 1 pack's 72 parent scales into the library.

        A curated entry wins.  It carries prayogas, resting notes, gamaka and
        a tempo range the pack has none of, so a melakarta already in
        ``raagas.json`` keeps everything it has and gains only what the pack
        knows that it does not: the chakra, the three blocks, their character,
        the starter tags and the suggested uses.  The pack's own spelling
        becomes an alias, so "Mechakalyani" and "Kalyani" reach one entry
        rather than two (specification sections 34 and 35: nothing curated is
        overwritten, and one thing is one thing).

        Records are matched by melakarta number, never by name.  Names would
        be guesswork here - "Bhairavi" is a janya and "Natabhairavi" is
        melakarta 20, and no amount of transliteration-matching makes that
        distinction safely.

        A melakarta nobody curated joins as a scale-only raaga: its scale and
        its block character, with everything else left unknown.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("cannot read melakarta file %s: %s", path, exc)
            return

        by_number: Dict[int, Raaga] = {}
        for raaga in self._raagas.values():
            if raaga.melakarta and raaga.melakarta not in by_number:
                by_number[int(raaga.melakarta)] = raaga

        merged = added = 0
        for entry in data.get("melakartas", []):
            try:
                number = int(entry["id"])
                blocks = dict(entry.get("block_character", {}))
                existing = by_number.get(number)
                if existing is not None:
                    if normalise_name(existing.name) not in normalise_name(entry["name"]) \
                            and normalise_name(entry["name"]) not in normalise_name(existing.name):
                        log.warning("melakarta %d is %r in the pack and %r in the "
                                    "library; keeping the library's name",
                                    number, entry["name"], existing.name)
                    existing.chakra = existing.chakra or entry.get("chakra", "")
                    existing.rg = existing.rg or entry.get("rg", "")
                    existing.madhyama = existing.madhyama or entry.get("madhyama", "")
                    existing.dn = existing.dn or entry.get("dn", "")
                    existing.block_character = existing.block_character or blocks
                    existing.tags = existing.tags or list(entry.get("tags", []))
                    existing.good_for = existing.good_for or list(entry.get("good_for", []))
                    if entry["name"].lower() != existing.name.lower() \
                            and entry["name"] not in existing.aliases:
                        existing.aliases.append(entry["name"])
                    self._index(existing, entry["name"])
                    merged += 1
                    continue

                raaga = Raaga(
                    name=entry["name"],
                    arohanam=list(entry.get("arohanam", [])),
                    avarohanam=list(entry.get("avarohanam", [])),
                    # Nothing else is known, and section 37 says unknown
                    # fields stay unknown: no jeeva, nyasa, prayogas, gamaka,
                    # moods, and no tempo range to pretend a preference with.
                    tempo_range=[],
                    melakarta=number,
                    chakra=entry.get("chakra", ""),
                    rg=entry.get("rg", ""),
                    madhyama=entry.get("madhyama", ""),
                    dn=entry.get("dn", ""),
                    block_character=blocks,
                    tags=list(entry.get("tags", [])),
                    good_for=list(entry.get("good_for", [])),
                    # No curated note either; ``character()`` speaks for it
                    # from the block map instead.
                    notes="",
                    source="melakarta-pack",
                )
            except Exception as exc:  # noqa: BLE001
                log.error("bad melakarta entry %r: %s", entry.get("name"), exc)
                continue
            key = raaga.name.lower()
            if key in self._raagas:
                log.warning("melakarta %d is called %r, which the library "
                            "already uses; leaving the existing entry alone",
                            raaga.melakarta, raaga.name)
                continue
            self._raagas[key] = raaga
            self._index(raaga, raaga.name)
            added += 1

        log.info("raaga library: %d melakartas merged, %d added as scale-only; "
                 "%d raagas in all", merged, added, len(self._raagas))

    # -- lookup ------------------------------------------------------------
    def names(self) -> List[str]:
        return sorted(r.name for r in self._raagas.values())

    def all(self) -> List[Raaga]:
        return [self._raagas[k] for k in sorted(self._raagas)]

    def get(self, name: str) -> Optional[Raaga]:
        if not name:
            return None
        key = self._alias.get(name.strip().lower())
        if key:
            return self._raagas.get(key)
        # No exact spelling, so fall back to a partial match.  With seventy-two
        # melakartas plenty of names contain other names - Mechakalyani
        # contains Kalyani, Natabhairavi contains Bhairavi - and "whichever
        # the dictionary yielded first" is not an answer.  Two directions,
        # each with its own idea of the best match:
        #
        #   a name inside the text  ->  the longest one, the most specific
        #                               thing the creator actually named;
        #   the text inside a name  ->  the shortest one, the nearest whole
        #                               name to what they typed.
        low = name.strip().lower()
        inside_text = [a for a in self._alias if a in low]
        if inside_text:
            best = min(inside_text, key=lambda a: (-len(a), self._alias[a]))
        else:
            inside_name = [a for a in self._alias if low in a]
            if not inside_name:
                return None
            best = min(inside_name, key=lambda a: (len(a), self._alias[a]))
        return self._raagas.get(self._alias[best])

    def require(self, name: str) -> Raaga:
        raaga = self.get(name)
        if raaga is None:
            raise KeyError(f"Unknown raaga: {name!r}")
        return raaga

    def find_in_text(self, text: str) -> Optional[Raaga]:
        """The raaga a creator named in free text, if they named one.

        Whole words only.  Bare substring matching was safe enough with
        eighteen raagas; with seventy-two melakartas in the library a short
        name buried inside an ordinary word would start answering briefs
        nobody wrote.  The longest match still wins, so "Mechakalyani" beats
        "Kalyani" and "Natabhairavi" beats "Bhairavi".
        """
        low = (text or "").lower()
        best: Optional[Raaga] = None
        best_len = 0
        for key, target in self._alias.items():
            if len(key) <= best_len:
                continue
            if re.search(rf"\b{re.escape(key)}\b", low):
                best_len = len(key)
                best = self._raagas.get(target)
        return best

    def by_mood(self, words: Sequence[str], limit: int = 5) -> List[Tuple[Raaga, float]]:
        scored = [(r, r.mood_score(words)) for r in self.all()]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0].name))
        return scored[:limit]

    def add_user_raaga(self, entry: dict) -> Raaga:
        path = config_dir() / USER_FILE_NAME
        data = {"raagas": []}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {"raagas": []}
        data.setdefault("raagas", [])
        data["raagas"] = [e for e in data["raagas"]
                         if e.get("name", "").lower() != entry.get("name", "").lower()]
        data["raagas"].append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.load(path, "user")
        return self.require(entry["name"])


_library: Optional[RaagaLibrary] = None


def library() -> RaagaLibrary:
    global _library
    if _library is None:
        _library = RaagaLibrary()
    return _library
