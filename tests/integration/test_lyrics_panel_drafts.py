"""Integration: an incomplete lyric draft is unmistakable in the panel.

Real controller, the real LyricsPanel offscreen, a stand-in writer.  What
the creator sees when a writer answers short: which version is approved,
which is a draft, how many lines are missing and where, who wrote each
line, and that viewing a draft does not approve it.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication            # noqa: E402

from raagacomposer.ui import theme                    # noqa: E402
from raagacomposer.ui.panels.lyrics_panel import LyricsPanel  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]

TAMIL_FIRST_LINE = "திரும்பி வந்தாய், அன்பே"


class _Writer:
    available = True
    name = "test-writer"

    def __init__(self, lines):
        self._lines = lines

    def write_lyrics(self, slots, brief):
        return list(self._lines)


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(theme.STYLESHEET)
    return application


def _a_tune(app, title):
    app.new_project(title, write=False)
    app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                     situation="A hopeful reunion after a long separation",
                     notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                           "Charanam and Ending.")
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=37)
    deadline = time.time() + 60
    while (app.project.melody() is None or app.jobs.active_jobs()) and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    assert app.project.melody() is not None, app.status_text


def _write(app, settle, writer):
    app.providers.llm = writer
    app.generate_lyrics(seed=4)
    settle(60)
    return app.project.lyrics[-1]


def _column(panel, header):
    for col in range(panel.table.columnCount()):
        if panel.table.horizontalHeaderItem(col).text() == header:
            return col
    raise AssertionError(f"no column {header!r}")


def test_a_short_answer_is_shown_as_an_incomplete_draft(app, settle, qt_app):
    _a_tune(app, "Draft visibility")
    from raagacomposer.lyrics.fitting import build_slots
    slots = build_slots(app.project.melody())
    approved = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * len(slots)))
    draft = _write(app, settle, _Writer([TAMIL_FIRST_LINE]))
    missing = len(slots) - 1
    assert draft.unfitted == missing

    panel = LyricsPanel(app)
    try:
        qt_app.processEvents()
        # The panel says, in words, what happened and what stands.
        standing = panel.standing.text()
        assert "draft" in standing.lower(), standing
        assert str(missing) in standing and "missing" in standing.lower(), standing
        assert f"v{approved.version}" in standing and "approved" in standing.lower(), standing
        assert "test-writer" in standing, standing
        # The version list tells the two apart.
        labels = [panel.versions.itemText(i) for i in range(panel.versions.count())]
        assert any("approved" in t.lower() and f"v{approved.version}" in t for t in labels), labels
        assert any("draft" in t.lower() and str(missing) in t and f"v{draft.version}" in t
                   for t in labels), labels

        # Viewing the draft does not approve it.
        panel.versions.setCurrentIndex(panel.versions.findData(draft.version))
        panel._version_chosen(panel.versions.currentIndex())
        qt_app.processEvents()
        assert app.project.approved_lyrics == approved.version
        assert panel.table.rowCount() == len(draft.lines)
        syll, by, line_col = (_column(panel, "Syllables"), _column(panel, "Written by"),
                              _column(panel, "Line"))
        assert panel.table.item(0, by).text() == "test-writer"
        assert panel.table.item(0, line_col).text() == TAMIL_FIRST_LINE
        for row in range(1, panel.table.rowCount()):
            assert "missing" in panel.table.item(row, syll).text().lower(), row
            assert "missing" in panel.table.item(row, by).text().lower(), row
        # The section that got no line is named where the creator reads it.
        assert slots[1].section_name in panel.alignment.toPlainText()

        # Accept on an incomplete draft is refused, and says why.
        panel.accept_btn.click()
        qt_app.processEvents()
        assert app.project.approved_lyrics == approved.version
        assert "missing" in app.status_text.lower(), app.status_text

        # Writing the missing lines into the draft makes it acceptable.
        panel._loading = False
        for row in range(1, panel.table.rowCount()):
            panel.table.item(row, line_col).setText("மலர்ந்தேன் நானே")
            qt_app.processEvents()
        assert draft.unfitted == 0, draft.notes
        assert all(l.source == "creator" for l in draft.lines[1:])
        assert "draft" not in panel.standing.text().lower() or "0 missing" in panel.standing.text()
        panel.accept_btn.click()
        qt_app.processEvents()
        assert app.project.approved_lyrics == draft.version
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


def test_a_complete_version_reads_as_approved_with_its_authors(app, settle, qt_app):
    _a_tune(app, "Complete version")
    from raagacomposer.lyrics.fitting import build_slots
    slots = build_slots(app.project.melody())
    lyrics = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * len(slots)))
    panel = LyricsPanel(app)
    try:
        qt_app.processEvents()
        standing = panel.standing.text()
        assert f"v{lyrics.version}" in standing and "approved" in standing.lower(), standing
        assert "missing" not in standing.lower()
        by = _column(panel, "Written by")
        assert all(panel.table.item(r, by).text() == "test-writer"
                   for r in range(panel.table.rowCount()))
        app.edit_lyric_line(lyrics.lines[0].id, "மலர்ந்தேன் நானே")
        panel.refresh()
        assert panel.table.item(0, by).text() == "creator"
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


# ----------------------------------------------------------------------
# Arya's independent review of 319d281: looking is never approving, and a
# line action reaches the version on the table, not the newest draft
# ----------------------------------------------------------------------
def _view(panel, qt_app, version):
    panel.versions.setCurrentIndex(panel.versions.findData(version))
    panel._version_chosen(panel.versions.currentIndex())
    qt_app.processEvents()


def test_browsing_a_repaired_but_unaccepted_draft_does_not_approve_it(app, settle, qt_app):
    _a_tune(app, "Repaired draft")
    from raagacomposer.lyrics.fitting import build_slots
    slots = build_slots(app.project.melody())
    approved = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * len(slots)))
    draft = _write(app, settle, _Writer([TAMIL_FIRST_LINE]))
    assert draft.unfitted == len(slots) - 1
    panel = LyricsPanel(app)
    try:
        qt_app.processEvents()
        _view(panel, qt_app, draft.version)
        line_col = _column(panel, "Line")
        panel._loading = False
        for row in range(1, panel.table.rowCount()):
            panel.table.item(row, line_col).setText("மலர்ந்தேன் நானே")
            qt_app.processEvents()
        assert draft.unfitted == 0, draft.notes
        assert app.project.approved_lyrics == approved.version, \
            "writing the missing lines is not accepting the draft"
        # Look at the approved words, then back at the repaired draft.
        _view(panel, qt_app, approved.version)
        assert app.project.approved_lyrics == approved.version
        _view(panel, qt_app, draft.version)
        assert app.project.approved_lyrics == approved.version, \
            "browsing a complete but unaccepted draft approved it"
        assert panel.shown().version == draft.version
        standing = panel.standing.text()
        assert f"v{approved.version} is approved" in standing, standing
        assert "Accept lyrics" in standing, standing
        # Accept is the one way.
        panel.accept_btn.click()
        qt_app.processEvents()
        assert app.project.approved_lyrics == draft.version
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


def test_unlocking_viewed_approved_line_does_not_change_another_draft(app, settle, qt_app):
    _a_tune(app, "Lock routed by version")
    from raagacomposer.lyrics.fitting import build_slots
    slots = build_slots(app.project.melody())
    first = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * len(slots)))
    assert app.project.approved_lyrics == first.version
    app.set_lyric_line_lock(first.lines[0].id, True)
    assert first.lines[0].locked
    # The next draft carries the locked line, id and all.
    second = _write(app, settle, _Writer(["மலர்ந்தேன் நானே"] * len(slots)))
    assert second.version != first.version
    carried = second.lines[0]
    assert carried.id == first.lines[0].id and carried.locked
    assert carried is not first.lines[0], "two versions share one line object"

    panel = LyricsPanel(app)
    try:
        qt_app.processEvents()
        _view(panel, qt_app, first.version)
        assert panel.shown().version == first.version
        panel.table.setCurrentCell(0, _column(panel, "Line"))
        panel._toggle_lock()
        qt_app.processEvents()
        assert not first.lines[0].locked, "the unlock did not reach the version on the table"
        assert second.lines[0].locked, "the unlock reached a draft that was not on the table"
        # And the edit goes to the version on the table too.
        panel._loading = False
        panel.table.item(0, _column(panel, "Line")).setText("வா வா அன்பே")
        qt_app.processEvents()
        assert first.lines[0].text == "வா வா அன்பே" and first.lines[0].source == "creator"
        assert second.lines[0].text == TAMIL_FIRST_LINE
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()
