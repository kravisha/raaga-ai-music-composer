"""The action status contract (v0.3 section 6.1, cross-referenced from the
old Master Specification's "visible action state" requirement).

Every important UI action must visibly move through

    Idle -> Starting -> Working -> Completed / Failed / Cancelled

and a failure must always carry a user-readable message, a diagnostic code a
person can quote back, and technical detail for the log - never "nothing
happened" (section 54).  Apply Brief (section 6) is the first action wired to
this, but the shape is deliberately generic so later actions - raaga
selection, tune generation, provider calls - can report through the same
contract without inventing their own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class ActionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (ActionState.COMPLETED, ActionState.FAILED,
                        ActionState.CANCELLED)


@dataclass
class ActionStatus:
    """One snapshot of an action's progress.

    ``phase`` is the human phrase shown while work is under way ("Analyzing
    creative brief..."); ``message`` is the sentence shown once the action has
    something final to say.  ``code`` is a short diagnostic reference such as
    ``"BRIEF-001"`` that a person can quote in a bug report or a log search;
    it is empty when nothing is wrong.  ``detail`` is the technical detail
    (typically ``repr(exc)``) that belongs in the log, not in front of the
    creator.
    """

    action: str
    state: ActionState
    phase: str = ""
    message: str = ""
    code: str = ""
    detail: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    # Which run of a background job produced this status.  ``target`` is the
    # JobManager target and ``epoch`` its epoch at submission; a status whose
    # epoch is no longer current for its target is stale and is dropped
    # rather than shown (section 53: stale output never overwrites newer
    # intent - the words in the status bar included).  Zero means "not from
    # a job", and such statuses are never stale.
    target: str = ""
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.state.terminal and not self.finished_at:
            self.finished_at = time.time()

    @property
    def text(self) -> str:
        """What a status label shows: the phase while working, otherwise the
        final message."""
        return self.phase or self.message
