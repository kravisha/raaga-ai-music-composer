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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir

log = get_logger("raaga.library")

DATA_FILE = Path(__file__).with_name("data") / "raagas.json"
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
        return (f"{self.name}\n"
                f"  Arohanam:   {' '.join(self.arohanam)}\n"
                f"  Avarohanam: {' '.join(self.avarohanam)}\n"
                f"  Jeeva swaras: {', '.join(self.jeeva) or '-'}\n"
                f"  Resting (nyasa): {', '.join(self.nyasa) or '-'}\n"
                f"  Moods: {', '.join(self.moods) or '-'}\n"
                f"  Tempo: {self.tempo_range[0]}-{self.tempo_range[-1]} bpm\n"
                f"  {self.notes}")


class RaagaLibrary:
    def __init__(self, extra_path: Optional[Path] = None) -> None:
        self._raagas: Dict[str, Raaga] = {}
        self._alias: Dict[str, str] = {}
        self.load(DATA_FILE, "builtin")
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
        low = name.strip().lower()
        for k, r in self._raagas.items():
            if low in k or k in low:
                return r
        return None

    def require(self, name: str) -> Raaga:
        raaga = self.get(name)
        if raaga is None:
            raise KeyError(f"Unknown raaga: {name!r}")
        return raaga

    def find_in_text(self, text: str) -> Optional[Raaga]:
        low = (text or "").lower()
        best: Optional[Raaga] = None
        best_len = 0
        for key, target in self._alias.items():
            if key in low and len(key) > best_len:
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
