"""Integration: the Tune panel shows what a directed rewrite did.

The landing status says it once and the tune render overwrites it; the
version keeps it as guidance_note, and the panel shows the note of the
version on show - including what was declined or could not be done -
and nothing for an ordinary version.  Offscreen, real controller.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from raagacomposer.core.models import SectionKind  # noqa: E402
from raagacomposer.music.direction import read_direction  # noqa: E402
from raagacomposer.ui.panels.tune_panel import TunePanel  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _settle(app, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.pump()
        if not app.jobs.active_jobs():
            app.pump()
            if not app.jobs.active_jobs():
                return
        time.sleep(0.02)
    raise TimeoutError([j.description for j in app.jobs.active_jobs()])


def _a_tune(app, title):
    app.new_project(title, write=False)
    app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                     situation="A hopeful reunion after a long separation",
                     notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                           "Charanam and Ending.")
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=37)
    _settle(app)
    assert app.project.melody() is not None, app.status_text


def test_the_panel_shows_the_direction_of_the_version_on_show(app, qt_app):
    _a_tune(app, "Direction on the panel")
    v1 = app.project.melody()
    charanam = next(s for s in v1.sections if s.kind is SectionKind.CHARANAM)
    panel = TunePanel(app)
    try:
        qt_app.processEvents()
        assert not panel.direction_note.isVisibleTo(panel) or panel.direction_note.text() == ""
        assert panel.direction_note.text() == ""

        app.regenerate_tune_section(charanam.id, read_direction("softer and plainer, faster"))
        _settle(app)
        v2 = app.project.melody()
        assert v2.version == v1.version + 1 and v2.guidance_note
        panel.refresh()
        qt_app.processEvents()
        text = panel.direction_note.text()
        assert text.startswith(f"v{v2.version}:"), text
        assert "softer" in text and "less gamaka" in text, text
        assert "not a control I have: faster" in text, text
        assert panel.direction_note.isVisibleTo(panel)

        # Pick the earlier version: nothing directed it, so nothing is said.
        panel.versions.setCurrentIndex(panel.versions.findData(v1.version))
        panel._version_chosen(panel.versions.currentIndex())
        panel.refresh()
        qt_app.processEvents()
        assert app.project.melody().version == v1.version
        assert panel.direction_note.text() == "" and not panel.direction_note.isVisibleTo(panel)

        # Back to the directed one, and it is said again.
        panel.versions.setCurrentIndex(panel.versions.findData(v2.version))
        panel._version_chosen(panel.versions.currentIndex())
        panel.refresh()
        qt_app.processEvents()
        assert "softer" in panel.direction_note.text()

        # A plain rewrite of the same section says nothing: no leak.
        charanam2 = next(s for s in app.project.melody().sections
                         if s.kind is SectionKind.CHARANAM)
        app.regenerate_tune_section(charanam2.id)
        _settle(app)
        panel.refresh()
        qt_app.processEvents()
        assert app.project.melody().version == v2.version + 1
        assert panel.direction_note.text() == ""
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


def test_a_note_with_only_a_refusal_is_still_shown(app, qt_app):
    """A version made from a direction whose controls could not be met
    (only the infeasible or declined part remains) still tells the creator
    what was asked and not done."""
    _a_tune(app, "Only a refusal")
    charanam = next(s for s in app.project.melody().sections if s.kind is SectionKind.CHARANAM)
    direction = read_direction("not softer, with more gamaka")
    app.regenerate_tune_section(charanam.id, direction)
    _settle(app)
    panel = TunePanel(app)
    try:
        qt_app.processEvents()
        text = panel.direction_note.text()
        assert "more gamaka" in text and "not applied, as asked: softer" in text, text
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()
