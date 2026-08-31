"""Curriculum engine (learning specification sections 3.2, 4, 5).

The curriculum is executable data, not documentation: every unit names a
practice handler and the parameters that configure it, and the engine decides
what to study next from prerequisites, thresholds and stored progress.

Stage A is universal ear training.  Stage B is instantiated per raaga from a
single template, so learning a second raaga costs no new curriculum data.
Stage C needs two raagas already taken to Stage B depth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from .knowledge import KnowledgeRepository, UnitProgress

log = get_logger("agent.curriculum")

DATA_FILE = Path(__file__).with_name("data") / "curriculum.json"
USER_FILE_NAME = "curriculum_user.json"

# A raaga counts as "known well enough" for cross-raaga work at this unit.
CROSS_RAAGA_GATE = "b11.drift"


@dataclass
class Unit:
    curriculum_unit_id: str
    level: int = 1
    stage: str = "A"
    skill_type: str = "generate.pattern"
    raaga_name: Optional[str] = None
    learning_goal: str = ""
    prerequisite_units: List[str] = field(default_factory=list)
    source_requirements: str = ""
    exercises: int = 5
    minimum_examples_required: int = 5
    minimum_pass_score: float = 0.7
    retry_policy: str = "retry_until_3_failures"
    params: Dict = field(default_factory=dict)
    scope: str = "universal"

    @property
    def id(self) -> str:
        """Unique id: Stage B units are per raaga."""
        if self.raaga_name:
            return f"{self.curriculum_unit_id}:{self.raaga_name}"
        return self.curriculum_unit_id

    def prerequisites(self) -> List[str]:
        """Prerequisite ids, qualified by raaga for Stage B units."""
        out = []
        for prereq in self.prerequisite_units:
            if self.raaga_name and prereq.startswith("b"):
                out.append(f"{prereq}:{self.raaga_name}")
            else:
                out.append(prereq)
        return out

    def max_failures(self) -> int:
        if self.retry_policy.startswith("retry_until_"):
            try:
                return int(self.retry_policy.split("_")[2])
            except (IndexError, ValueError):
                return 3
        return 3

    def describe(self) -> str:
        where = f" [{self.raaga_name}]" if self.raaga_name else ""
        return f"{self.id}{where}: {self.learning_goal}"


class CurriculumEngine:
    def __init__(self, repository: KnowledgeRepository,
                 data_file: Optional[Path] = None,
                 pilot_raaga: str = "Keeravani",
                 revisit_after: float = 180.0) -> None:
        self.repo = repository
        self.pilot_raaga = pilot_raaga
        # How long a rested lesson waits before it is offered again, and how
        # many attempts it gets in total before it is genuinely set aside.
        self.revisit_after = float(revisit_after)
        self.max_attempts_per_unit = 12
        self._universal: List[Unit] = []
        self._template: List[Unit] = []
        self._cross: List[Unit] = []
        self.version = 0
        self.load(Path(data_file) if data_file else DATA_FILE)
        user_file = config_dir() / USER_FILE_NAME
        if user_file.exists():
            self.load(user_file, merge=True)

    # -- loading -----------------------------------------------------------
    def load(self, path: Path, merge: bool = False) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("cannot read curriculum %s: %s", path, exc)
            return
        self.version = max(self.version, int(data.get("curriculum_version", 1)))

        def build(entries: Sequence[dict], scope: str) -> List[Unit]:
            out = []
            for entry in entries:
                fields = {k: v for k, v in entry.items()
                          if k in Unit.__dataclass_fields__}
                fields.setdefault("scope", scope)
                out.append(Unit(**fields))
            return out

        universal = build(data.get("units", []), "universal")
        template = build(data.get("raaga_template", []), "per_raaga")
        cross = build(data.get("cross_raaga", []), "cross")

        if merge:
            self._universal = _merge(self._universal, universal)
            self._template = _merge(self._template, template)
            self._cross = _merge(self._cross, cross)
        else:
            self._universal, self._template, self._cross = universal, template, cross
        log.info("curriculum v%d: %d universal, %d per-raaga, %d cross-raaga",
                 self.version, len(self._universal), len(self._template),
                 len(self._cross))

    # -- unit access -------------------------------------------------------
    def universal_units(self) -> List[Unit]:
        return list(self._universal)

    def raaga_units(self, raaga: str) -> List[Unit]:
        out = []
        for unit in self._template:
            clone = Unit(**{f: getattr(unit, f) for f in Unit.__dataclass_fields__})
            clone.raaga_name = raaga
            out.append(clone)
        return out

    def cross_units(self) -> List[Unit]:
        return list(self._cross)

    def all_units(self, raagas: Optional[Sequence[str]] = None) -> List[Unit]:
        raagas = list(raagas) if raagas else [self.current_raaga()]
        units = self.universal_units()
        for raaga in raagas:
            units.extend(self.raaga_units(raaga))
        units.extend(self.cross_units())
        return units

    def unit(self, unit_id: str) -> Optional[Unit]:
        base, _, raaga = unit_id.partition(":")
        for candidate in self._universal + self._cross:
            if candidate.curriculum_unit_id == base:
                return candidate
        if raaga:
            for candidate in self.raaga_units(raaga):
                if candidate.curriculum_unit_id == base:
                    return candidate
        return None

    # -- state -------------------------------------------------------------
    def current_raaga(self) -> str:
        return self.repo.state("current_raaga", self.pilot_raaga)

    def set_current_raaga(self, raaga: str) -> None:
        self.repo.set_state("current_raaga", raaga)
        self.repo.log_event("curriculum.raaga_changed", raaga, raaga=raaga)

    def mastered_raagas(self) -> List[str]:
        """Raagas taken at least as far as the cross-raaga gate."""
        passed = set(self.repo.completed_units())
        out = []
        for unit_id in passed:
            base, _, raaga = unit_id.partition(":")
            if base == CROSS_RAAGA_GATE and raaga:
                out.append(raaga)
        return sorted(set(out))

    def is_passed(self, unit: Unit) -> bool:
        return self.repo.progress(unit.id).status == "passed"

    def blocked_by(self, unit: Unit) -> List[str]:
        """Prerequisites that are not yet passed."""
        passed = set(self.repo.completed_units())
        missing = [p for p in unit.prerequisites() if p not in passed]
        needed = int(unit.params.get("requires_mastered_raagas", 0))
        if needed and len(self.mastered_raagas()) < needed:
            missing.append(f"{needed} raagas mastered "
                           f"({len(self.mastered_raagas())} so far)")
        return missing

    def is_available(self, unit: Unit) -> bool:
        return not self.blocked_by(unit)

    # -- scheduling --------------------------------------------------------
    def next_unit(self, raaga: Optional[str] = None) -> Optional[Unit]:
        """The next unit to work on: earliest level that is unblocked and unpassed."""
        raaga = raaga or self.current_raaga()
        candidates = self.universal_units() + self.raaga_units(raaga)
        if len(self.mastered_raagas()) >= 2:
            candidates += self.cross_units()

        now = time.time()
        available: List[Unit] = []
        rested: List[Tuple[float, Unit]] = []
        for unit in candidates:
            progress = self.repo.progress(unit.id)
            if progress.status == "passed":
                continue
            if not self.is_available(unit):
                continue
            if progress.status == "failed" and \
                    progress.failures >= unit.max_failures():
                # A rested lesson is revisited later, not abandoned: a student
                # who cannot do something today may manage it next week.  With
                # nothing else to do, it is picked up again straight away -
                # sitting idle teaches nobody anything.
                due = now - progress.last_attempted_at >= self.revisit_after
                if (due or progress.attempts < self.max_attempts_per_unit):
                    rested.append((progress.mastery, unit))
                continue
            available.append(unit)

        if available:
            available.sort(key=lambda u: (u.level, u.curriculum_unit_id))
            return available[0]
        if rested:
            rested.sort(key=lambda item: -item[0])
            unit = rested[0][1]
            progress = self.repo.progress(unit.id)
            progress.failures = 0
            progress.status = "in_progress"
            progress.notes = "revisiting after a rest"
            self.repo.save_progress(progress)
            self.repo.log_event("curriculum.revisit", unit.id, unit_id=unit.id,
                                raaga=unit.raaga_name or "")
            return unit
        return None

    def units_needing_sources(self, raaga: Optional[str] = None) -> List[Unit]:
        """Units that cannot pass until more material has been ingested."""
        raaga = raaga or self.current_raaga()
        return [u for u in self.raaga_units(raaga)
                if u.retry_policy == "ingest_more_sources"
                and self.repo.progress(u.id).status != "passed"]

    # -- recording results -------------------------------------------------
    def record_attempt(self, unit: Unit, score: float, passed: bool,
                       note: str = "") -> UnitProgress:
        progress = self.repo.progress(unit.id)
        progress.raaga = unit.raaga_name or ""
        progress.attempts += 1
        progress.mastery = max(progress.mastery, round(float(score), 4))
        progress.last_attempted_at = __import__("time").time()
        progress.notes = note[:400]
        if passed:
            progress.status = "passed"
            progress.completed_at = progress.last_attempted_at
        else:
            progress.failures += 1
            progress.status = ("failed"
                               if progress.failures >= unit.max_failures()
                               else "in_progress")
        self.repo.save_progress(progress)
        self.repo.log_event(
            "curriculum.attempt",
            f"{'passed' if passed else 'failed'} score={score:.2f} {note[:120]}",
            unit_id=unit.id, raaga=unit.raaga_name or "")
        return progress

    def reset_unit(self, unit_id: str) -> None:
        progress = self.repo.progress(unit_id)
        progress.status = "not_started"
        progress.failures = 0
        progress.mastery = 0.0
        self.repo.save_progress(progress)

    # -- reporting ---------------------------------------------------------
    def stage_summary(self, raaga: Optional[str] = None) -> Dict[str, object]:
        raaga = raaga or self.current_raaga()
        universal = self.universal_units()
        per_raaga = self.raaga_units(raaga)
        passed = set(self.repo.completed_units())

        def done(units: Sequence[Unit]) -> int:
            return sum(1 for u in units if u.id in passed)

        next_unit = self.next_unit(raaga)
        total = len(universal) + len(per_raaga)
        completed = done(universal) + done(per_raaga)
        return {
            "curriculum_version": self.version,
            "current_raaga": raaga,
            "stage": ("A" if done(universal) < len(universal)
                      else ("C" if len(self.mastered_raagas()) >= 2 else "B")),
            "foundations": f"{done(universal)}/{len(universal)}",
            "raaga_units": f"{done(per_raaga)}/{len(per_raaga)}",
            "cross_units": f"{done(self.cross_units())}/{len(self.cross_units())}",
            "overall_percent": round(100.0 * completed / max(1, total), 1),
            "mastered_raagas": self.mastered_raagas(),
            "next_unit": next_unit.id if next_unit else "",
            "next_goal": next_unit.learning_goal if next_unit else
                         "Curriculum complete for this raaga.",
        }

    def progress_table(self, raaga: Optional[str] = None) -> List[Dict[str, object]]:
        raaga = raaga or self.current_raaga()
        rows = []
        for unit in self.universal_units() + self.raaga_units(raaga) + \
                self.cross_units():
            progress = self.repo.progress(unit.id)
            rows.append({
                "unit": unit.id,
                "level": unit.level,
                "stage": unit.stage,
                "goal": unit.learning_goal,
                "status": progress.status,
                "mastery": round(progress.mastery, 2),
                "attempts": progress.attempts,
                "blocked_by": ", ".join(self.blocked_by(unit)),
            })
        return rows


def _merge(base: List[Unit], extra: List[Unit]) -> List[Unit]:
    by_id = {u.curriculum_unit_id: u for u in base}
    for unit in extra:
        by_id[unit.curriculum_unit_id] = unit
    return sorted(by_id.values(), key=lambda u: (u.level, u.curriculum_unit_id))
