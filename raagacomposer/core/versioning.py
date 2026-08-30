"""Undo/redo and lock protection (spec sections 12.28, 16).

Undo works on whole-project snapshots.  That is heavier than a command log but
it cannot desynchronise, which matters more here: the creator must never lose
an accepted tune because an undo entry was written incorrectly.  Audio files on
disk are never deleted by an undo, so stepping back and forward again always
finds its artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .logging_setup import get_logger
from .models import ApprovalState, Project, Region, Section, Track
from .serde import from_jsonable, to_jsonable

log = get_logger("versioning")


class LockedContentError(Exception):
    """Raised when an operation would modify content the creator locked."""


@dataclass
class Snapshot:
    label: str
    data: dict


class UndoManager:
    def __init__(self, depth: int = 60) -> None:
        self.depth = depth
        self._stack: List[Snapshot] = []
        self._index = -1

    def reset(self, project: Project, label: str = "opened") -> None:
        self._stack = [Snapshot(label, to_jsonable(project))]
        self._index = 0

    def commit(self, project: Project, label: str) -> None:
        if self._index < len(self._stack) - 1:
            del self._stack[self._index + 1:]
        self._stack.append(Snapshot(label, to_jsonable(project)))
        if len(self._stack) > self.depth:
            del self._stack[0]
        self._index = len(self._stack) - 1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._stack) - 1

    def undo_label(self) -> str:
        return self._stack[self._index].label if self.can_undo else ""

    def redo_label(self) -> str:
        return self._stack[self._index + 1].label if self.can_redo else ""

    def undo(self) -> Optional[Tuple[Project, str]]:
        if not self.can_undo:
            return None
        label = self._stack[self._index].label
        self._index -= 1
        return from_jsonable(Project, self._stack[self._index].data), label

    def redo(self) -> Optional[Tuple[Project, str]]:
        if not self.can_redo:
            return None
        self._index += 1
        snap = self._stack[self._index]
        return from_jsonable(Project, snap.data), snap.label


# --------------------------------------------------------------------------
# Lock protection
# --------------------------------------------------------------------------
def locked_sections_in(project: Project, start: float, end: float) -> List[Section]:
    melody = project.melody()
    if not melody:
        return []
    return [s for s in melody.sections
            if s.locked and s.start < end and start < s.end]


def locked_regions_in(track: Track, start: float, end: float) -> List[Region]:
    return [r for r in track.regions if r.locked and r.overlaps(start, end)]


def assert_unlocked_track(track: Track, start: float, end: float,
                          what: str = "this region") -> None:
    if track.locked:
        raise LockedContentError(
            f"Track '{track.label}' is locked. Unlock it before changing {what}.")
    clashes = locked_regions_in(track, start, end)
    if clashes:
        span = ", ".join(f"{r.start:.0f}-{r.end:.0f}s" for r in clashes)
        raise LockedContentError(
            f"'{track.label}' has locked regions at {span}. Unlock them first.")


def assert_melody_editable(project: Project, start: float, end: float) -> None:
    melody = project.melody()
    if melody is None:
        return
    if melody.state == ApprovalState.LOCKED and not _covers_whole(melody, start, end):
        # Locking the tune protects it wholesale; section locks are finer.
        raise LockedContentError(
            "The tune is locked. Unlock it to regenerate part of the melody.")
    clashes = locked_sections_in(project, start, end)
    if clashes:
        names = ", ".join(s.name for s in clashes)
        raise LockedContentError(
            f"Locked section(s): {names}. Unlock before regenerating that range.")


def _covers_whole(melody, start: float, end: float) -> bool:
    return start <= 0.0 and end >= melody.duration - 1e-6
