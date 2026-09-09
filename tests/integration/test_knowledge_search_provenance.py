"""Integration: a new Knowledge search never keeps the last result's source.

Offscreen TrainingPanel on the real controller with fixture knowledge
entries; no provider, no training run.  Arya's independent panel review
of fae3ae8: searching Alpha, selecting it, then searching Beta showed
Beta's row beside Alpha's source; a search with no result kept Alpha's
source beside an empty table.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from raagacomposer.training.models import KnowledgeEntry  # noqa: E402
from raagacomposer.ui.panels.training_panel import TrainingPanel  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, qt_app):
    app.new_project("Knowledge search", write=False)
    # The training store outlives one test in the shared test home, so the
    # fixture names carry a tag that makes each test's entries its own.
    import uuid
    tag = uuid.uuid4().hex[:6]
    for name, raaga in ((f"Alpha{tag}", "Hamsadhwani"), (f"Beta{tag}", "Keeravani")):
        app.training.store.add_knowledge(KnowledgeEntry(
            subject=raaga, raga=raaga, normalized_statement=f"{name} fixture statement",
            category="scale", source_title=f"{name} fixture source",
            source_id=f"fixture-{name}", source_timestamp="00:10",
            evidence=f"{name} fixture evidence", confidence=0.61,
            run_id=f"fixture-run-{name}"))
    widget = TrainingPanel(app)
    widget._timer.stop()
    widget.alpha, widget.beta = f"Alpha{tag}", f"Beta{tag}"
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        qt_app.processEvents()


def _table_text(table):
    return "\n".join(table.item(r, c).text() for r in range(table.rowCount())
                     for c in range(table.columnCount()) if table.item(r, c))


def _search(panel, qt_app, text):
    panel.knowledge_query.setText(text)
    panel.knowledge_query.returnPressed.emit()
    qt_app.processEvents()


def test_replacing_search_results_cannot_keep_previous_source(panel, qt_app):
    _search(panel, qt_app, panel.alpha)
    assert panel.knowledge_table.rowCount() == 1
    panel.knowledge_table.setCurrentCell(0, 0)
    qt_app.processEvents()
    assert f"{panel.alpha} fixture source" in panel.provenance_view.toPlainText()

    _search(panel, qt_app, panel.beta)
    assert f"{panel.beta} fixture statement" in _table_text(panel.knowledge_table)
    assert panel.alpha not in _table_text(panel.knowledge_table)
    assert f"{panel.alpha} fixture source" not in panel.provenance_view.toPlainText(), \
        "Beta's row is shown beside Alpha's source"
    assert panel.knowledge_table.currentRow() < 0, "the old selection survived the new search"
    # Selecting the new row binds the new entry's source, explicitly.
    panel.knowledge_table.setCurrentCell(0, 0)
    qt_app.processEvents()
    assert f"{panel.beta} fixture source" in panel.provenance_view.toPlainText()


def test_empty_knowledge_search_clears_unrelated_provenance(panel, qt_app):
    _search(panel, qt_app, panel.alpha)
    panel.knowledge_table.setCurrentCell(0, 0)
    qt_app.processEvents()
    assert f"{panel.alpha} fixture source" in panel.provenance_view.toPlainText()
    _search(panel, qt_app, "no-such-fixture-entry")
    assert panel.knowledge_table.rowCount() == 0
    assert panel.provenance_view.toPlainText() == "", \
        "an empty search kept the source of a result no longer shown"
    assert "0 knowledge item" in panel.status.text()


def test_an_explicit_selection_still_shows_its_own_source(panel, qt_app):
    for name in (panel.alpha, panel.beta):
        _search(panel, qt_app, name)
        assert panel.knowledge_table.rowCount() == 1
        panel.knowledge_table.setCurrentCell(0, 0)
        qt_app.processEvents()
        assert f"{name} fixture source" in panel.provenance_view.toPlainText(), name
