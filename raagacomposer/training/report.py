"""Rendering a Learning Report, and looking back over them - sections 8, 12.

The report itself is built by the pipeline and stored by the store; this turns
one into something a person reads.  The section order is the specification's,
and two of its rules are structural rather than cosmetic:

* "What I understood" and "What I learned" are separate headings, always, even
  when one of them is empty.  Section 20 rule 5.  A source that was understood
  but taught nothing new should look like that on the page.
* a source whose content was never analysed says so at the top, in the
  specification's own words, before anything that might read like a finding.

Nothing here invents content.  If a section has nothing in it, it says so.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from .models import (Accessibility, LearningReport, ObjectiveStatus,
                     RunStatus)
from .store import TrainingStore

log = get_logger("training.report")

METADATA_ONLY_NOTICE = "METADATA ONLY - CONTENT NOT ANALYZED"


def _when(stamp: float) -> str:
    if not stamp:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


def _bullets(items: Sequence[str], empty: str) -> List[str]:
    return [f"  - {item}" for item in items] if items else [f"  ({empty})"]


class LearningReportService:
    """Builds, stores and renders reports."""

    def __init__(self, store: TrainingStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    def report(self, run_id: str) -> Optional[LearningReport]:
        return self.store.report(run_id)

    def render(self, report: LearningReport) -> str:
        """The whole of section 8, as plain text."""
        source = report.source
        run = self.store.run(report.run_id)
        lines: List[str] = []

        def heading(text: str) -> None:
            lines.extend(["", text, "-" * len(text)])

        lines.append("LEARNING REPORT")
        lines.append("=" * 15)
        if report.analysed_representation == "none":
            lines.extend(["", METADATA_ONLY_NOTICE])

        # -- 8.1 source information -------------------------------------
        heading("Source")
        if source is not None:
            access_label = Accessibility.LABELS.get(
                source.accessibility_status, source.accessibility_status)
            lines.extend([
                f"  Title          {source.title}",
                f"  URL            {source.url or '-'}",
                f"  Source         {source.provider or source.source_type}",
                f"  Author         {source.author or '-'}",
                f"  Duration       {source.duration_label}",
                f"  Accessibility  {access_label}",
            ])
        lines.extend([
            f"  Processed      {_when(report.generated_at)}",
            f"  Found by       "
            f"{(run.search_phrase if run else '') or '-'}",
            f"  Analysed       {report.analysed_representation or 'nothing'}",
        ])

        # -- 8.2 objectives ---------------------------------------------
        heading(f"Learning objectives ({report.objectives_met} of "
                f"{len(report.objectives)} met)")
        if not report.objectives:
            lines.append("  (none were set)")
        for objective in report.objectives:
            label = ObjectiveStatus.LABELS.get(objective.status,
                                               objective.status)
            lines.append(f"  [{label}] {objective.description}")
            if objective.outcome:
                lines.append(f"      result     {objective.outcome}")
            if objective.evidence:
                lines.append(f"      evidence   {objective.evidence}")
            lines.append(f"      confidence {objective.confidence:.2f}")

        # -- 8.3 summary -------------------------------------------------
        heading("Summary")
        lines.append(f"  {report.summary or '(nothing to summarise)'}")

        # -- 8.4 and 8.5, kept apart -------------------------------------
        heading("What I understood")
        lines.append(f"  {report.understood or '(nothing was understood)'}")

        heading("What I learned")
        lines.extend(_bullets(report.learned, "nothing new"))

        # -- 8.6 -----------------------------------------------------------
        heading("Existing knowledge confirmed")
        lines.extend(_bullets(report.confirmed,
                              "nothing already held was reinforced"))

        # -- 8.7 -----------------------------------------------------------
        heading("Conflicts and disagreements")
        if not report.conflicts:
            lines.append("  (none)")
        for conflict in report.conflicts:
            resolved = ("yes" if conflict.resolved
                        else "not yet - nothing has been overwritten")
            lines.extend([
                f"  Existing   {conflict.existing_claim}",
                f"    held at  {conflict.existing_confidence:.2f}",
                f"  New        {conflict.new_claim}",
                f"    at       {conflict.new_confidence:.2f}",
                f"    evidence {conflict.source_evidence or '-'}",
                f"  Suggested  {conflict.recommendation}",
                f"  Resolved   {resolved}",
                "",
            ])

        # -- 8.8 -----------------------------------------------------------
        heading("Practical application")
        lines.extend(_bullets(report.practical_application,
                              "nothing here changes what is played"))

        # -- 8.9 -----------------------------------------------------------
        heading("Confidence")
        lines.append(f"  {report.confidence_band()} "
                     f"({report.confidence:.2f}) across "
                     f"{len(report.learned)} learned item(s)")

        # -- honesty, which is not optional --------------------------------
        heading("What this does not tell you")
        lines.extend(_bullets(report.honest_limits, "no reservations recorded"))

        # -- 8.10 ----------------------------------------------------------
        heading("Recommended next learning")
        lines.extend(_bullets(report.next_learning, "nothing outstanding"))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def render_run(self, run_id: str) -> str:
        report = self.store.report(run_id)
        if report is None:
            return "No report has been produced for this source yet."
        return self.render(report)


class TrainingHistoryService:
    """Section 12 - what has been learned, and from what."""

    def __init__(self, store: TrainingStore) -> None:
        self.store = store

    def entries(self, *, raga: str = "", status: str = "", topic: str = "",
                min_confidence: float = 0.0, since: float = 0.0,
                limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.store.history(raga=raga, status=status,
                                  min_confidence=min_confidence, since=since,
                                  phrase=topic, limit=limit)
        for row in rows:
            row["status_label"] = RunStatus.LABELS.get(row["status"],
                                                       row["status"])
            row["when"] = _when(row["completed_at"])
        return rows

    def totals(self) -> Dict[str, Any]:
        runs = self.store.runs(limit=1000)
        return {
            "sources_seen": len(runs),
            "completed": sum(1 for r in runs
                             if r.status == RunStatus.COMPLETED),
            "failed": sum(1 for r in runs
                          if r.status in (RunStatus.FAILED,
                                          RunStatus.SOURCE_INACCESSIBLE)),
            "queued": sum(1 for r in runs if r.status == RunStatus.QUEUED),
            "knowledge_items": self.store.knowledge_count(),
            "open_conflicts": len(self.store.conflicts(unresolved_only=True)),
        }
