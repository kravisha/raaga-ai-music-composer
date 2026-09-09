"""Integration: a request that names a section is about that section.

"Write the words for the Pallavi", "sing the Pallavi", "give me a variation
of the Charanam": the section parser already found the section in each of
these, and the controller then acted on the whole song.  Real controller,
a stand-in writer where words are involved, no model, no device.
"""
from __future__ import annotations

import time

import pytest

from raagacomposer.core.models import SectionKind

pytestmark = pytest.mark.integration

TAMIL = "திரும்பி வந்தாய், அன்பே"


class _Writer:
    available = True
    name = "test-writer"

    def __init__(self, line):
        self.line = line
        self.asked = []

    def write_lyrics(self, slots, brief):
        self.asked.append(len(slots))
        return [self.line] * len(slots)


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


def _settle(app, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.pump()
        if not app.jobs.active_jobs():
            app.pump()
            if not app.jobs.active_jobs():
                return
        time.sleep(0.02)
    raise TimeoutError([j.description for j in app.jobs.active_jobs()])


def _section(app, kind):
    return next(s for s in app.project.melody().sections if s.kind is kind)


def _notes_of(melody, section_id):
    return [(n.midi, round(n.start, 3), round(n.duration, 3))
            for n in melody.notes if n.section_id == section_id]


def _lines_by_section(lyrics):
    out = {}
    for line in lyrics.lines:
        out.setdefault(line.section_id, []).append(line.text)
    return out


# ----------------------------------------------------------------------
def test_words_asked_for_one_section_are_written_for_that_section_only(app):
    _a_tune(app, "Words for the Pallavi")
    app.providers.llm = _Writer(TAMIL)
    app.handle_utterance("write the words")
    _settle(app)
    whole = app.project.lyrics[-1]
    before = _lines_by_section(whole)
    pallavi = _section(app, SectionKind.PALLAVI)
    assert len(before[pallavi.id]) >= 1 and len(before) >= 2

    app.providers.llm = writer = _Writer("மலர்ந்தேன் நானே")
    cmd = app.handle_utterance("write the words for the Pallavi")
    assert cmd.intent == "lyrics.generate" and cmd.section_id == pallavi.id, cmd
    _settle(app)
    after = _lines_by_section(app.project.lyrics[-1])
    assert app.project.lyrics[-1].version == whole.version + 1
    assert all(t == "மலர்ந்தேன் நானே" for t in after[pallavi.id]), after[pallavi.id]
    for section_id, lines in before.items():
        if section_id != pallavi.id:
            assert after[section_id] == lines, "another section's words were rewritten"
    assert writer.asked == [len(before[pallavi.id])], \
        "the writer was asked for the whole song, not the Pallavi"


def test_words_asked_for_a_locked_section_are_refused_by_name(app):
    _a_tune(app, "Locked Pallavi words")
    pallavi = _section(app, SectionKind.PALLAVI)
    app.set_section_lock(pallavi.id, True)
    app.providers.llm = writer = _Writer(TAMIL)
    versions = len(app.project.lyrics)
    app.handle_utterance("write the words for the Pallavi")
    _settle(app)
    assert len(app.project.lyrics) == versions
    assert writer.asked == []
    assert "Pallavi is locked" in app.status_text, app.status_text


def test_sing_the_section_sings_that_section_not_the_song(app, monkeypatch):
    _a_tune(app, "Sing the Pallavi")
    app.providers.llm = _Writer(TAMIL)
    app.handle_utterance("write the words")
    _settle(app)
    pallavi = _section(app, SectionKind.PALLAVI)
    asked = []
    monkeypatch.setattr(app, "preview_section",
                        lambda section_id, autoplay=True: asked.append(("section", section_id)))
    monkeypatch.setattr(app, "render_vocal",
                        lambda *a, **k: asked.append(("whole", k.get("section_ids"))))
    cmd = app.handle_utterance("sing the Pallavi")
    assert cmd.intent == "voice.render" and cmd.section_id == pallavi.id, cmd
    assert asked == [("section", pallavi.id)], asked
    asked.clear()
    app.handle_utterance("sing it")
    assert asked == [("whole", None)], asked


def test_a_variation_of_one_section_leaves_the_others_where_they_were(app):
    _a_tune(app, "Vary the Charanam")
    v1 = app.project.melody()
    charanam = _section(app, SectionKind.CHARANAM)
    cmd = app.handle_utterance("give me a variation of the Charanam")
    assert cmd.intent == "tune.variation" and cmd.section_id == charanam.id, cmd
    _settle(app)
    v2 = app.project.melody()
    assert v2.version == v1.version + 1, app.status_text
    changed = [s.name for s in v2.sections
               if _notes_of(v2, s.id) != _notes_of(v1, next(
                   o for o in v1.sections if o.name == s.name).id)]
    assert changed == ["Charanam 1"], changed


def test_a_variation_of_a_locked_section_is_refused_and_nothing_moves(app):
    _a_tune(app, "Vary a locked Charanam")
    v1 = app.project.melody()
    charanam = _section(app, SectionKind.CHARANAM)
    app.set_section_lock(charanam.id, True)
    app.handle_utterance("give me a variation of the Charanam")
    _settle(app)
    assert app.project.melody() is v1
    assert "locked" in app.status_text.lower(), app.status_text


# ----------------------------------------------------------------------
# The property asked of a section rides with the rewrite (section direction)
# ----------------------------------------------------------------------
def _section_notes(melody, kind):
    section = next(s for s in melody.sections if s.kind is kind)
    return [n for n in melody.notes if n.section_id == section.id]


def _mean_velocity(notes):
    return sum(n.velocity for n in notes) / max(1, len(notes))


def test_make_the_section_softer_softens_that_section_only(app):
    _a_tune(app, "Softer Charanam")
    v1 = app.project.melody()
    charanam = _section(app, SectionKind.CHARANAM)
    cmd = app.handle_utterance("make the Charanam softer and plainer")
    assert cmd.intent == "tune.regenerate_section" and cmd.section_id == charanam.id, cmd
    _settle(app)
    v2 = app.project.melody()
    assert v2.version == v1.version + 1, app.status_text
    changed = [s.name for s in v2.sections
               if _notes_of(v2, s.id) != _notes_of(v1, next(
                   o for o in v1.sections if o.name == s.name).id)]
    assert changed == ["Charanam 1"], changed
    before = _section_notes(v1, SectionKind.CHARANAM)
    after = _section_notes(v2, SectionKind.CHARANAM)
    assert _mean_velocity(after) < _mean_velocity(before)
    # (The gamaka count is a draw, not a bound - the fixed-seed unit test
    # in test_melody_direction.py is where "plainer" is measured; the
    # rewrite here takes a time-based seed.)
    # The landing line names the controls (the status moves on when the
    # tune render lands, so it is read from the history it was filed in).
    landed = [h.description for h in app.project.history if h.action == "tune.version"][-1]
    assert "softer" in landed and "less gamaka" in landed, landed
    assert "softer" in v2.guidance_note and "less gamaka" in v2.guidance_note
    # No leakage: the next plain rewrite of the same section starts plain.
    app.handle_utterance("rewrite the Charanam")
    _settle(app)
    v3 = app.project.melody()
    assert v3.version == v2.version + 1 and v3.guidance_note == ""
    assert _mean_velocity(_section_notes(v3, SectionKind.CHARANAM)) > _mean_velocity(after)


def _frozen(app):
    """Everything a refused request must leave alone."""
    m = app.project.melody()
    return (len(app.project.melodies), m.version,
            [(n.swara, n.midi, n.start) for n in m.notes],
            len(app.project.history), len(app.project.lyrics))


@pytest.mark.parametrize("text,said", [
    ("make the Charanam faster", "not a control I have: faster"),
    ("rewrite the Charanam faster", "not a control I have: faster"),
    ("make the Charanam not softer", "not applied, as asked: softer"),
    ("make the Charanam softer and stronger", "asked both ways, so neither: softer and stronger"),
])
def test_a_direction_with_nothing_to_do_rewrites_nothing_and_says_why(app, text, said):
    """Arya's review of fae3ae8: each of these added a version and changed
    the Charanam while saying the one thing asked was not done.  Now the
    tune, its version, its notes and the history stay, and the reason is
    the reply."""
    _a_tune(app, "Nothing to do")
    before = _frozen(app)
    cmd = app.handle_utterance(text)
    assert cmd.intent == "tune.regenerate_section", cmd
    _settle(app)
    assert _frozen(app)[:3] == before[:3], text
    assert "Charanam 1 left as it is" in app.status_text and said in app.status_text, app.status_text
    # the conversation records the turn, and nothing else was written
    assert len(app.project.history) == before[3]


def test_a_plain_rewrite_and_an_explicit_variation_still_make_a_version(app):
    _a_tune(app, "Plain rewrite still works")
    v1 = app.project.melody()
    app.handle_utterance("rewrite the Charanam")
    _settle(app)
    assert app.project.melody().version == v1.version + 1
    app.handle_utterance("give me a variation of the Charanam")
    _settle(app)
    assert app.project.melody().version == v1.version + 2


@pytest.mark.parametrize("text", [
    "keep the Pallavi unchanged; make the Charanam softer",
    "make the Charanam softer; keep the Pallavi unchanged",
    "leave the Pallavi alone and make the Charanam softer",
])
def test_a_kept_section_is_kept_and_the_change_lands_on_the_other(app, text):
    """Arya's review of fae3ae8: the first-named section was rewritten -
    the Pallavi, the one asked to be kept - and the Charanam left alone.
    The sentence is now placed by clause, whichever order it comes in."""
    _a_tune(app, "Keep one, change the other")
    v1 = app.project.melody()
    pallavi = _section(app, SectionKind.PALLAVI)
    charanam = _section(app, SectionKind.CHARANAM)
    app.handle_utterance(text)
    _settle(app)
    v2 = app.project.melody()
    assert v2.version == v1.version + 1, app.status_text
    changed = [s.name for s in v2.sections
               if _notes_of(v2, s.id) != _notes_of(v1, next(
                   o for o in v1.sections if o.name == s.name).id)]
    assert changed == ["Charanam 1"], (text, changed)
    assert _notes_of(v2, pallavi.id) == _notes_of(v1, pallavi.id)
    assert not pallavi.locked
    assert _mean_velocity(_section_notes(v2, SectionKind.CHARANAM)) < \
        _mean_velocity(_section_notes(v1, SectionKind.CHARANAM))
    assert "softer" in v2.guidance_note and charanam.name in [s.name for s in v2.sections]


def test_the_command_and_the_turn_describe_the_section_actually_rewritten(app):
    """Arya's P2 on fe9fb3e: the Charanam was rewritten, but the command
    still said Pallavi - section, time, remembered target and the turn's
    action all read the first section spoken of.  A refused request must
    not read as a rewrite either."""
    _a_tune(app, "Metadata follows the deed")
    pallavi = _section(app, SectionKind.PALLAVI)
    charanam = _section(app, SectionKind.CHARANAM)
    cmd = app.handle_utterance("keep the Pallavi unchanged; make the Charanam softer")
    _settle(app)
    assert cmd.section_id == charanam.id, cmd
    assert cmd.time is not None and cmd.time.section_id == charanam.id
    assert cmd.time.start == pytest.approx(charanam.start) and cmd.time.end == pytest.approx(charanam.end)
    assert app.context.last_section_id == charanam.id
    turn = app.project.conversation[-1]
    assert turn.status == "applied"
    assert "Charanam" in turn.action and "Pallavi" not in turn.action, turn.action
    assert turn.action.startswith("Rewrite the section"), turn.action
    # What was understood reads the same as what was done - on the turn,
    # on the command, and on the Conversation panel's "Understood" line.
    assert turn.interpretation == turn.action, (turn.interpretation, turn.action)
    assert cmd.interpretation == turn.action
    assert "Pallavi" not in turn.interpretation
    _panel_says(app, understood=turn.action, result="Completed")
    # ("that section" / "this part" are the playhead's or the selection's
    # section in the time parser, not the remembered one; what is checked
    # here is that the remembered one is the Charanam, as asserted above.)

    # A refused request: not a rewrite, not remembered.
    before_last = app.context.last_section_id
    cmd = app.handle_utterance("rewrite the Pallavi but keep the Pallavi unchanged")
    _settle(app)
    assert cmd.section_id == "" and cmd.time is None
    turn = app.project.conversation[-1]
    assert turn.status == "declined", turn.status
    assert turn.action.startswith("Nothing changed"), turn.action
    assert "kept, as you asked" in turn.reason
    assert app.context.last_section_id == before_last != pallavi.id
    assert turn.interpretation == turn.action and "Pallavi" not in turn.interpretation
    _panel_says(app, understood=turn.action, result="Declined")


def _panel_says(app, understood: str, result: str) -> None:
    """The offscreen Conversation panel's Understood and Result lines for
    the last turn."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from raagacomposer.ui.panels.conversation_panel import ConversationPanel
    qt_app = QApplication.instance() or QApplication([])
    panel = ConversationPanel(app)
    try:
        panel.refresh()
        qt_app.processEvents()
        assert panel.understood_label.text() == understood, panel.understood_label.text()
        assert panel.result_label.text().startswith(result), panel.result_label.text()
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


def test_a_sentence_that_only_keeps_or_locks_rewrites_nothing(app):
    _a_tune(app, "Only kept")
    before = _frozen(app)
    app.handle_utterance("keep the Pallavi unchanged")
    _settle(app)
    assert _frozen(app)[:3] == before[:3]
    # "keep ..." is not a rewrite verb, so it may not even be a section
    # command; when it is, nothing moves.  The explicit form is refused
    # with the reason.
    app.handle_utterance("rewrite the Pallavi but keep the Pallavi unchanged")
    _settle(app)
    assert _frozen(app)[:3] == before[:3]
    assert "kept, as you asked" in app.status_text, app.status_text
    charanam = _section(app, SectionKind.CHARANAM)
    app.set_section_lock(charanam.id, True)
    app.handle_utterance("make the Charanam softer")
    _settle(app)
    assert _frozen(app)[:3] == before[:3]
    assert "locked" in app.status_text.lower(), app.status_text


def test_two_sections_to_change_in_one_breath_are_refused_before_anything_moves(app):
    _a_tune(app, "Two at once")
    before = _frozen(app)
    app.handle_utterance("make the Pallavi softer and make the Charanam plainer")
    _settle(app)
    assert _frozen(app)[:3] == before[:3]
    assert "One section at a time" in app.status_text, app.status_text


def test_a_direction_on_a_locked_section_is_refused(app):
    _a_tune(app, "Locked and softer")
    v1 = app.project.melody()
    charanam = _section(app, SectionKind.CHARANAM)
    app.set_section_lock(charanam.id, True)
    app.handle_utterance("make the Charanam softer")
    _settle(app)
    assert app.project.melody() is v1
    assert "locked" in app.status_text.lower(), app.status_text


def test_a_fader_request_naming_a_section_is_still_a_fader_request(app):
    from raagacomposer.speech.intent import interpret
    _a_tune(app, "Fader in the Pallavi")
    app._sync_context()
    ctx = app.context.time_context()
    assert interpret("turn the violin down in the Pallavi", ctx).intent == "arrange.level"
    assert interpret("make the Pallavi softer", ctx).intent == "tune.regenerate_section"
    assert interpret("change the violin to veena in the Pallavi", ctx).intent == "arrange.replace"


def test_a_variation_with_no_section_named_is_of_the_whole_tune(app):
    _a_tune(app, "Vary it all")
    v1 = app.project.melody()
    cmd = app.handle_utterance("give me a variation")
    assert cmd.intent == "tune.variation" and not cmd.section_id
    _settle(app)
    v2 = app.project.melody()
    assert v2.version == v1.version + 1 and "variation" in (v2.derived_from or "").lower()
