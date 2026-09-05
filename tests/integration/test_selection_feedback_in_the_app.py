"""Selection feedback through the real controller - plan item S3.

The unit tests prove the weights behave; these prove the application sends
the right signals at the right moments, which is where every increment in
this project has actually gone wrong.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from raagacomposer.core.models import CreativeBrief

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])

SAD = dict(situation="love failure", mood="sad romantic",
           feel="lonely late at night but still warm")


def _apply(app, **fields):
    app.apply_brief_sync(**fields)
    return [s.name for s in app.last_suggestions or []]


def test_choosing_a_suggestion_teaches_the_agent(app):
    offered = _apply(app, **SAD)
    assert len(offered) >= 2
    app.select_raaga(offered[1])

    weights = {(w.raaga, w.dimension): w for w in app.agent.selection_preferences()}
    chosen = [w for (raaga, _), w in weights.items() if raaga == offered[1]]
    assert chosen and all(w.weight > 0 for w in chosen)
    # ... and the one it was taken ahead of was noticed, more gently.
    passed = [w for (raaga, _), w in weights.items() if raaga == offered[0]]
    assert passed and all(w.weight < 0 for w in passed)
    assert max(w.weight for w in chosen) > abs(min(w.weight for w in passed))


def test_the_application_choosing_for_itself_teaches_nothing(app):
    """``require_raaga`` picks when the creator has not.  Learning a
    preference from that is the agent learning its own habits back."""
    _apply(app, **SAD)
    app.require_raaga()
    assert app.agent.selection_preferences() == []


def test_rejecting_a_suggestion_lowers_it_next_time(app):
    before = _apply(app, **SAD)
    correction = app.reject_raaga(before[0], "too sad")
    assert correction.get("sadness", 0.0) < 0

    after = _apply(app, **SAD)
    assert after[0] != before[0]
    assert before[0] not in after[:2]


def test_a_rejection_for_one_feeling_leaves_another_alone(app):
    joyful = dict(situation="a wedding", mood="joyful", feel="bright and festive")
    joyful_before = _apply(app, **joyful)
    sad_before = _apply(app, **SAD)
    app.reject_raaga(sad_before[0], "too sad")
    assert _apply(app, **joyful) == joyful_before


def test_what_was_learned_can_be_shown_and_withdrawn(app):
    offered = _apply(app, **SAD)
    app.reject_raaga(offered[0], "too sad")
    assert app.agent.selection_preferences()
    assert any(offered[0] in w.describe()
               for w in app.agent.selection_preferences())

    app.agent.forget_selection_preferences(offered[0])
    assert app.agent.selection_preferences(offered[0]) == []
    assert _apply(app, **SAD)[0] == offered[0], "withdrawing restores the ranking"


def test_preferences_outlive_the_application(app, settings):
    from raagacomposer.app import AppController

    before = _apply(app, **SAD)
    app.reject_raaga(before[0], "too sad")
    learned = {(w.raaga, w.dimension): round(w.weight, 4)
               for w in app.agent.selection_preferences()}
    assert learned
    app.close()

    restarted = AppController(settings)
    try:
        again = {(w.raaga, w.dimension): round(w.weight, 4)
                 for w in restarted.agent.selection_preferences()}
        assert again == learned
        restarted.update_brief(**SAD)
        suggestions = restarted.agent.suggest_raagas(restarted.project.brief, 4)
        assert [s.name for s in suggestions][0] != before[0]
    finally:
        restarted.close()


def test_feedback_is_about_the_brief_the_suggestions_answered(app):
    """Found by running it, not by a test.

    The choice has to be attached to the brief it was answering.  Attaching
    it to whatever happened to be in the panel taught the agent that a raaga
    chosen for a grieving brief suits a wedding.
    """
    sad = _apply(app, **SAD)
    app.select_raaga(sad[1])
    dimensions = {w.dimension for w in app.agent.selection_preferences(sad[1])}
    assert "sadness" in dimensions
    assert "joy" not in dimensions and "brightness" not in dimensions


def test_a_raaga_we_did_not_suggest_teaches_nothing(app):
    """An override is not a preference.

    Apply one brief, apply a second, then name a raaga from the first list:
    we no longer know which feeling the creator had in mind, and a guess is
    what produced the defect above.  The selection still works - it is only
    the learning that stands down.
    """
    sad = _apply(app, **SAD)
    _apply(app, situation="a wedding", mood="joyful", feel="bright and festive")
    assert sad[0] not in (app.project.raaga.alternatives or [])

    app.select_raaga(sad[0])
    assert app.project.raaga.selected == sad[0]
    assert app.agent.selection_preferences(sad[0]) == []
    assert app.reject_raaga(sad[0], "too sad") == {}


@pytest.mark.ui
def test_the_panel_can_turn_a_suggestion_down(app, qt_app):
    """Without a control for it the creator could only ever teach the agent
    by agreeing with it."""
    from raagacomposer.ui.panels.raaga_panel import RaagaPanel

    _apply(app, **SAD)
    panel = RaagaPanel(app)
    panel.suggest()                      # populates the list, as the app does
    panel.suggestions.setCurrentRow(0)
    rejected = panel._current_name()
    assert rejected in (app.project.raaga.alternatives or [])

    panel.reject_selected()

    assert app.agent.selection_preferences(rejected)
    shown = [panel.suggestions.item(i).data(Qt.UserRole)
             for i in range(panel.suggestions.count())]
    assert shown and shown[0] != rejected


def test_feedback_never_changes_what_the_raaga_is(app):
    """The pack's rule, at the level a creator would notice it."""
    offered = _apply(app, **SAD)
    raaga = app.raagas.require(offered[0])
    before = (list(raaga.arohanam), list(raaga.avarohanam), list(raaga.jeeva),
              list(raaga.nyasa))
    for _ in range(5):
        app.reject_raaga(offered[0], "too sad")
    after = (list(raaga.arohanam), list(raaga.avarohanam), list(raaga.jeeva),
             list(raaga.nyasa))
    assert after == before


# --------------------------------------------------------------------------
# audition (pack document 05 section 7, plan item S4)
# --------------------------------------------------------------------------
def test_auditioning_renders_the_scale_and_can_be_played(app):
    _apply(app, **SAD)
    app.select_raaga((app.project.raaga.alternatives or [])[0])

    heard = app.audition_raaga(play=False)
    assert heard is not None
    assert len(heard.ascending) >= 5 and len(heard.descending) >= 5
    # It is real audio, of about the length the plan says it should be.
    rendered = app._renders.get("audition")
    assert rendered is not None
    assert rendered.duration == pytest.approx(heard.seconds, abs=0.5)


def test_auditioning_is_a_weaker_signal_than_choosing(app):
    """The pack's own weights: auditioned +0.2, accepted +1.0.  Listening is
    not yet agreeing."""
    offered = _apply(app, **SAD)
    app.audition_raaga(offered[1], play=False)
    after_hearing = max(w.weight for w in app.agent.selection_preferences(offered[1]))

    app.agent.forget_selection_preferences()
    app.select_raaga(offered[1])
    after_choosing = max(w.weight for w in app.agent.selection_preferences(offered[1]))

    assert 0 < after_hearing < after_choosing


def test_auditioning_a_raaga_we_did_not_suggest_still_plays_it(app):
    """Hearing a raaga is always allowed; only the learning needs to know
    which brief it answers."""
    _apply(app, **SAD)
    _apply(app, situation="a wedding", mood="joyful", feel="bright and festive")
    heard = app.audition_raaga("Shubhapantuvarali", play=False)
    assert heard is not None and heard.raaga == "Shubhapantuvarali"
    assert app.agent.selection_preferences("Shubhapantuvarali") == []


def test_auditioning_without_a_raaga_says_so_rather_than_failing(app):
    assert app.audition_raaga("no-such-raaga", play=False) is None


@pytest.mark.ui
def test_the_panel_can_play_the_scale(app, qt_app):
    from raagacomposer.ui.panels.raaga_panel import RaagaPanel

    _apply(app, **SAD)
    panel = RaagaPanel(app)
    panel.suggest()
    panel.suggestions.setCurrentRow(0)
    panel.audition_selected()
    assert app._renders.get("audition") is not None
