"""Regression: Apply Brief must never do nothing (v0.3 sections 6, 6.1).

Before this fix, ``BriefPanel.apply()`` only called ``app.update_brief`` and
emitted a Qt signal - the brief was saved but nothing suggested a raaga, and
nothing told the creator that.  A second click of a *different* button
("Suggest from the brief", ``RaagaPanel.suggest()``) was the only way to see
a suggestion, and even then there was no progress and no diagnostic on
failure.  These tests guard the fix: ``AppController.apply_brief`` /
``apply_brief_sync`` validate, report Starting/Working/Completed/Failed
through ``on_action`` (the action status contract, ``core/actions.py``), and
always produce either ranked suggestions or an explicit, coded failure.

TEST A from the canonical specification (section 63) is
``test_apply_brief_sync_reports_progress_and_suggests_raagas`` below.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from raagacomposer.core.actions import ActionState
from raagacomposer.core.models import CreativeBrief, Project
from raagacomposer.core.persistence import PROJECT_FILE, ProjectStore
from raagacomposer.core.serde import to_jsonable

pytestmark = pytest.mark.regression


def _collect(app):
    """Attach a collector to ``app.on_action`` without clobbering anything
    already listening, and return the list it appends to."""
    seen = []
    previous = app.on_action

    def _on_action(status):
        if previous:
            previous(status)
        seen.append(status)

    app.on_action = _on_action
    return seen


# --------------------------------------------------------------------------
# TEST A - Apply Brief bug (spec section 63)
# --------------------------------------------------------------------------
def test_apply_brief_sync_reports_progress_and_suggests_raagas(app):
    seen = _collect(app)

    status = app.apply_brief_sync(
        situation="love failure", mood="sad",
        feel="lonely late at night but still warm")

    assert status.state == ActionState.COMPLETED
    states = [s.state for s in seen]
    assert states[0] == ActionState.STARTING
    assert states[-1] == ActionState.COMPLETED

    working_phases = [s.phase for s in seen if s.state == ActionState.WORKING]
    expected = ["Analyzing creative brief...",
               "Searching learned raga knowledge...",
               "Ranking suggestions..."]
    # Every expected phase appears, in order (extra phases - e.g. a fallback
    # warning - may be interleaved, but never reordered).
    positions = [working_phases.index(p) for p in expected]
    assert positions == sorted(positions)

    suggestions = app.last_suggestions
    assert suggestions, "Apply Brief must never leave the panel empty"
    for s in suggestions:
        assert s.name
        assert getattr(s, "reason", "")
        assert 0.0 <= float(s.confidence) <= 1.0
    assert app.project.raaga.alternatives == [s.name for s in suggestions]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_apply_brief_never_fails_silently_on_an_empty_brief(app):
    before = len(app.project.errors)

    status = app.apply_brief_sync(situation="", mood="", feel="")

    assert status.state == ActionState.FAILED
    assert status.code == "BRIEF-001"
    assert status.message
    assert len(app.project.errors) == before + 1
    assert app.project.errors[-1].message == status.message


# --------------------------------------------------------------------------
# agent failure -> fallback, but never swallowed
# --------------------------------------------------------------------------
def test_apply_brief_falls_back_when_the_agent_fails(app, monkeypatch):
    """Choice documented here (queue item 2 left this open): a failure of the
    *agent's* ranking is recoverable - the rule engine in raaga/selection.py
    always answers - so Apply Brief still completes for the creator rather
    than blocking the whole action on one optional component.  The failure is
    not swallowed: it is logged with a traceback, reported as a WORKING phase
    naming the agent, and recorded in ``project.errors`` with a diagnostic
    code (BRIEF-003), same as section 54 asks of a "Provider failure".  BRIEF-
    002 stays reserved for a failure that leaves no suggestions to fall back
    on at all (see ``test_apply_brief_never_fails_silently_on_an_empty_brief``
    and the pipeline's outer exception handler in ``app.py``).
    """
    def boom(*args, **kwargs):
        raise RuntimeError("memory unavailable")

    monkeypatch.setattr(app.agent, "suggest_raagas", boom)
    seen = _collect(app)
    before = len(app.project.errors)

    status = app.apply_brief_sync(situation="a wedding at dawn",
                                  mood="celebration", feel="bright and warm")

    assert status.state == ActionState.COMPLETED
    assert app.last_suggestions, "the rule engine must still answer"

    warnings = [s for s in seen if s.code == "BRIEF-003"]
    assert warnings, "the agent's failure must be visible, not swallowed"
    assert "agent" in warnings[0].phase.lower()
    assert len(app.project.errors) > before
    assert "agent" in app.project.errors[-1].message.lower()


# --------------------------------------------------------------------------
# the asynchronous path (spec section 53: never block the GUI thread)
# --------------------------------------------------------------------------
def test_apply_brief_async_delivers_the_final_status_via_pump(app, settle):
    seen = []
    calling_thread = threading.current_thread()

    def _on_action(status):
        seen.append((status, threading.current_thread()))

    app.on_action = _on_action

    started = app.apply_brief(situation="a long journey home", mood="hopeful",
                              feel="tired but hopeful")
    assert started.state == ActionState.STARTING

    settle()

    assert seen, "on_action must have been called at least once"
    for status, thread in seen:
        assert thread is calling_thread, \
            "action statuses from a background job must arrive on the " \
            "thread that pumps them, never the worker thread"
    assert seen[-1][0].state == ActionState.COMPLETED
    assert app.last_suggestions
    assert app.project.raaga.alternatives == [s.name for s in app.last_suggestions]


def test_a_second_apply_brief_supersedes_the_first(app, settle):
    """Section 53: a newer creator instruction wins - including the words
    shown for it.  No status from the first run may reach the UI after the
    second run has started, or the status bar would say "3 raagas
    suggested" for a brief the creator has already replaced."""
    seen = _collect(app)
    app.apply_brief(situation="sad", mood="sad", feel="sad")
    status = app.apply_brief(situation="joyful wedding", mood="celebration",
                             feel="bright and festive")
    assert status.state == ActionState.STARTING
    settle()
    final = app.actions["apply_brief"]
    assert final.state == ActionState.COMPLETED
    assert app.project.brief.situation == "joyful wedding"

    newest = status.epoch
    assert newest > 0
    after_second_start = seen[seen.index(status):]
    stale = [s for s in after_second_start if s.epoch and s.epoch < newest]
    assert not stale, [(s.state, s.text) for s in stale]
    assert final.epoch == newest
    assert not any(s.state == ActionState.CANCELLED for s in seen), \
        "a superseded run is not a cancellation the creator asked for"


def test_cancelling_apply_brief_reports_cancelled(app, settle, monkeypatch):
    """Section 6.1 names Cancelled as a state the creator must see."""
    import time as _time

    real = app.agent.suggest_raagas

    def slow(brief, limit):
        _time.sleep(0.4)
        return real(brief, limit)

    monkeypatch.setattr(app.agent, "suggest_raagas", slow)
    seen = _collect(app)
    app.apply_brief(situation="a slow one", mood="sad", feel="slow")
    app.jobs.cancel_all("cancelled by the creator")
    settle()
    final = app.actions["apply_brief"]
    assert final.state == ActionState.CANCELLED, [(s.state, s.text) for s in seen]
    assert app.last_suggestions == [], "a cancelled run must not store results"
    assert not any(s.state == ActionState.COMPLETED for s in seen)


# --------------------------------------------------------------------------
# song title (spec section 5) and backward compatibility
# --------------------------------------------------------------------------
def test_applying_the_brief_sets_the_project_title_from_it(app):
    app.apply_brief_sync(title="Terrace at Midnight", situation="love failure",
                         mood="sad", feel="lonely")
    assert app.project.title == "Terrace at Midnight"
    assert app.project.brief.title == "Terrace at Midnight"


def test_an_empty_brief_title_does_not_blank_the_project_title(app):
    app.project.title = "Kept Title"
    app.apply_brief_sync(title="", situation="love failure", mood="sad",
                         feel="lonely")
    assert app.project.title == "Kept Title"


def test_old_project_json_without_a_brief_title_still_loads(tmp_path: Path,
                                                             settings):
    """A ``project.json`` written before this fix has no ``brief.title`` key
    at all. ``from_jsonable`` must tolerate the missing key rather than
    raising, and the brief must load with the new field defaulted."""
    project = Project(title="Old Song")
    project.brief = CreativeBrief(situation="an old brief", mood="nostalgia")
    payload = to_jsonable(project)
    del payload["brief"]["title"]  # simulate a pre-v0.3 file

    store = ProjectStore(settings)
    directory = tmp_path / "old-project"
    store.ensure_dirs(directory)
    import json
    (directory / PROJECT_FILE).write_text(json.dumps(payload),
                                          encoding="utf-8")

    reopened = store.open(directory)
    assert reopened.brief.title == ""
    assert reopened.brief.situation == "an old brief"
