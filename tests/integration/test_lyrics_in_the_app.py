"""Integration: what the application does with words it cannot sing, and
whose words they are.  Real controller, a stand-in writer, no model."""
import time

from raagacomposer.core.models import SectionKind

TAMIL_FIRST_LINE = "திரும்பி வந்தாய், அன்பே"


class _Writer:
    available = True
    name = "test-writer"

    def __init__(self, lines):
        self._lines = lines

    def write_lyrics(self, slots, brief):
        return list(self._lines)


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


def test_a_writer_s_tamil_is_approved_and_carries_its_name(app, settle):
    _a_tune(app, "Tamil words from a writer")
    from raagacomposer.lyrics.fitting import build_slots
    count = len(build_slots(app.project.melody()))
    lyrics = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * count))
    assert app.project.approved_lyrics == lyrics.version
    assert [l.text for l in lyrics.lines] == [TAMIL_FIRST_LINE] * count
    assert all(l.source == "llm:test-writer" for l in lyrics.lines)
    assert lyrics.unfitted == 0
    assert "fitted" in app.status_text


def test_a_draft_with_unsingable_lines_is_kept_but_not_approved(app, settle):
    _a_tune(app, "A line with nothing to sing")
    from raagacomposer.lyrics.fitting import build_slots
    count = len(build_slots(app.project.melody()))
    first = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * count))
    assert app.project.approved_lyrics == first.version

    draft = _write(app, settle, _Writer(["..."] + [TAMIL_FIRST_LINE] * (count - 1)))
    assert draft.unfitted == 1
    assert draft.lines[0].text == "..." and draft.lines[0].syllables == []
    assert app.project.approved_lyrics == first.version, \
        "an unfitted draft must not replace the approved words"
    assert "draft" in app.status_text and "unfitted" in app.status_text, app.status_text
    # And the words that did fit were not swapped for anything else.
    assert draft.lines[1].text == TAMIL_FIRST_LINE


def test_a_line_the_creator_types_is_the_creator_s(app, settle):
    _a_tune(app, "Creator's own line")
    from raagacomposer.lyrics.fitting import build_slots
    count = len(build_slots(app.project.melody()))
    lyrics = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * count))
    line = lyrics.lines[0]
    assert line.source == "llm:test-writer"
    app.edit_lyric_line(line.id, "மலர்ந்தேன் நானே")
    assert line.text == "மலர்ந்தேன் நானே" and line.source == "creator"
    assert line.syllables and all(not s.isascii() for s in line.syllables if not s.startswith("~"))


def test_a_writer_s_blank_answer_does_not_replace_the_approved_words(app, settle):
    """The gate looked only at lines with text; an empty answer for a
    requested slot walked past it and replaced approved words."""
    _a_tune(app, "Blank answer")
    from raagacomposer.lyrics.fitting import build_slots
    count = len(build_slots(app.project.melody()))
    first = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * count))
    assert app.project.approved_lyrics == first.version
    for blank in ("", "   "):
        draft = _write(app, settle, _Writer([blank] + [TAMIL_FIRST_LINE] * (count - 1)))
        assert draft.unfitted == 1, draft.notes
        assert app.project.approved_lyrics == first.version, \
            "a blank answer replaced the approved words"
        assert "draft" in app.status_text, app.status_text


def test_a_partial_answer_keeps_the_approved_words_and_says_what_is_missing(app, settle):
    _a_tune(app, "Partial answer")
    from raagacomposer.lyrics.fitting import build_slots
    slots = build_slots(app.project.melody())
    first = _write(app, settle, _Writer([TAMIL_FIRST_LINE] * len(slots)))
    assert app.project.approved_lyrics == first.version
    approved_text = [l.text for l in first.lines]

    draft = _write(app, settle, _Writer([TAMIL_FIRST_LINE]))
    assert draft.unfitted == len(slots) - 1, draft.notes
    assert draft.lines[0].text == TAMIL_FIRST_LINE
    assert all(l.text == "" and l.source == "missing:test-writer" for l in draft.lines[1:])
    assert not any(l.source == "lexicon" for l in draft.lines)
    assert app.project.approved_lyrics == first.version
    assert [l.text for l in app.project.lyrics_version(first.version).lines] == approved_text
    assert "draft" in app.status_text and str(len(slots) - 1) in app.status_text, app.status_text
    for slot in slots[1:]:
        assert slot.section_name in draft.notes
