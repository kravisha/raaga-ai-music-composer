"""The song team's simple path (raagacomposer/studio, docs/SONG_TEAM.md).

Mood -> lyrics and tune at the same time -> beat -> the words fitted to the
phrase.  Each agent is tested against what it hands on, because the card is
the contract: any agent that writes the same card can replace it.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from raagacomposer.lyrics.fitting import build_slots, count_syllables
from raagacomposer.music import tala as tala_module
from raagacomposer.raaga import selection
from raagacomposer.studio import (
    LYRICIST,
    MOOD_LISTENER,
    PRODUCER,
    TALA_KEEPER,
    TUNESMITH,
    Lyricist,
    MoodListener,
    SongTeam,
    TalaKeeper,
)
from raagacomposer.studio.team import LINES, alapana_sketch, split_to_targets

pytestmark = pytest.mark.unit

LONELY = "lonely, late at night, but still warm"
LULLABY = "a lullaby for my daughter, gentle and sleepy"


@pytest.fixture(scope="module")
def team():
    return SongTeam()


@pytest.fixture(scope="module")
def lonely(team):
    return team.run(LONELY, "Tamil", seed=11)


@pytest.fixture(scope="module")
def lullaby(team):
    return team.run(LULLABY, "English", seed=11)


# --------------------------------------------------------------------------
# mood listener
# --------------------------------------------------------------------------
def test_the_mood_listener_hears_the_words_and_nothing_else():
    card = MoodListener().listen(LONELY, "Tamil")
    assert "lonely" in card.feel_words or "sad" in card.feel_words
    assert card.pace == "slow"
    assert len(card.line_plan) == LINES
    # A new project's default situation is a nervous novice; carrying it in
    # would make every song the team writes nervous.
    assert card.brief.situation == ""
    assert card.brief.mood == LONELY


def test_the_mood_listener_takes_a_language_named_in_the_mood():
    assert MoodListener().listen("a happy telugu wedding song",
                                 "Tamil").language == "Telugu"
    assert MoodListener().listen("quiet rain", "Hindi").language == "Hindi"


def test_an_empty_mood_is_answered_not_guessed():
    with pytest.raises(ValueError):
        MoodListener().listen("   ")


# --------------------------------------------------------------------------
# lyricist
# --------------------------------------------------------------------------
class _FakeModel:
    available = True
    name = "fake-model"

    def __init__(self, lines=None, fail=False):
        self.lines = lines or []
        self.fail = fail
        self.asked = None

    def write_lyrics(self, slots, brief):
        self.asked = [s.syllable_count for s in slots]
        if self.fail:
            raise RuntimeError("model is down")
        return list(self.lines)


def test_the_lyricist_writes_two_singable_lines_with_no_model():
    sheet = Lyricist().write(MoodListener().listen(LONELY), seed=3)
    assert len(sheet.lines) == LINES
    assert all(count_syllables(line) > 0 for line in sheet.lines)
    assert sheet.written_by == "built-in word lists"


def test_the_lyricist_asks_a_model_for_the_agreed_line_plan():
    mood = MoodListener().listen(LONELY, "English")
    model = _FakeModel(["the night is long and warm", "I wait for you"])
    sheet = Lyricist(model).write(mood)
    assert model.asked == mood.line_plan
    assert sheet.lines == ["the night is long and warm", "I wait for you"]
    assert sheet.written_by == "fake-model"


def test_an_unsingable_model_line_is_replaced_and_said_so():
    mood = MoodListener().listen(LONELY, "Tamil")
    sheet = Lyricist(_FakeModel(["இரவு நீளம்", "kaadhal nilavu"])).write(mood)
    assert count_syllables(sheet.lines[0]) > 0          # replaced
    assert sheet.lines[1] == "kaadhal nilavu"           # kept
    assert "word lists" in sheet.written_by


def test_a_failing_model_never_blocks_the_lyricist():
    sheet = Lyricist(_FakeModel(fail=True)).write(MoodListener().listen(LONELY))
    assert len(sheet.lines) == LINES
    assert sheet.written_by == "built-in word lists"


def test_a_rewrite_keeps_the_lyricists_own_words_where_they_fit():
    mood = MoodListener().listen(LONELY, "English")
    parts = Lyricist().revise("lonely moonlight", [2, 3], mood, seed=1)
    assert [count_syllables(p) for p in parts] == [2, 3]
    assert parts[0] == "lonely"
    assert parts[1].startswith("moonlight")


def test_split_to_targets_only_splits_where_the_words_fall():
    assert split_to_targets("kaadhal nilavu en", [2, 4]) == ["kaadhal",
                                                             "nilavu en"]
    assert split_to_targets("kaadhal nilavu", [3, 2]) is None
    assert split_to_targets("kaadhal nilavu en", [2, 3]) is None


# --------------------------------------------------------------------------
# tunesmith
# --------------------------------------------------------------------------
def test_the_tunesmith_takes_the_raaga_the_selection_engine_ranks_first(
        lonely, raagas):
    top = selection.suggest(lonely.mood.brief, raagas, limit=4)[0]
    assert lonely.tune.raaga == top.name
    assert lonely.tune.chosen_by == "tunesmith"
    assert lonely.tune.reason
    assert next(name for name, _ in lonely.tune.alternatives) == top.name


def test_every_note_of_the_phrase_and_the_alapana_is_in_the_raaga(lonely,
                                                                  raagas):
    raaga = raagas.require(lonely.tune.raaga)
    for note in lonely.tune.melody.notes + lonely.tune.alapana:
        assert raaga.is_allowed(note.swara), note.swara


def test_the_alapana_settles_on_sa_and_dwells_on_the_raagas_own_swaras(
        raagas):
    raaga = raagas.require("Mohanam")
    sketch = alapana_sketch(raaga, 60, seed=4)
    assert sketch[0].swara == "S" and sketch[-1].swara == "S"
    longest = max(sketch[1:-1], key=lambda n: n.duration)
    assert longest.swara in set(raaga.jeeva) | set(raaga.nyasa)
    # Unmetered, and more than a held note.
    assert len({n.swara for n in sketch}) >= 4
    assert len({round(n.duration, 2) for n in sketch}) > 2


def test_no_raaga_in_the_library_gets_a_sketch_that_goes_nowhere(raagas):
    # Pentatonic raagas once turned back at Ga: S R2 G3 R2 S.
    for raaga in raagas.all():
        for seed in range(4):
            swaras = [n.swara for n in alapana_sketch(raaga, 60, seed)]
            assert len(set(swaras)) >= 4, (raaga.name, swaras)
            assert swaras[-2:] != ["S", "S"], (raaga.name, swaras)


def test_the_producer_can_name_the_raaga(team):
    draft = team.run(LONELY, "Tamil", raaga="Mohanam", seed=2)
    assert draft.tune.raaga == "Mohanam"
    assert draft.tune.chosen_by == "producer"
    with pytest.raises(KeyError):
        team.run(LONELY, "Tamil", raaga="Not A Raaga", seed=2)


def test_the_phrase_is_two_lines(lonely):
    assert len(lonely.tune.phrase_lines) == LINES
    assert all(line.strip() for line in lonely.tune.phrase_lines)


# --------------------------------------------------------------------------
# both at once
# --------------------------------------------------------------------------
def test_the_lyricist_and_the_tunesmith_work_at_the_same_time():
    team = SongTeam()
    spans = {}
    real_write, real_propose = team.lyricist.write, team.tunesmith.propose

    def timed(name, fn):
        def run(*args, **kwargs):
            start = time.perf_counter()
            time.sleep(0.25)
            out = fn(*args, **kwargs)
            spans[name] = (start, time.perf_counter(),
                           threading.current_thread().name)
            return out
        return run

    team.lyricist.write = timed("lyrics", real_write)
    team.tunesmith.propose = timed("tune", real_propose)
    team.run(LONELY, "Tamil", seed=5)
    (l_start, l_end, l_thread) = spans["lyrics"]
    (t_start, t_end, t_thread) = spans["tune"]
    assert l_thread != t_thread
    assert l_start < t_end and t_start < l_end, "they ran one after the other"


# --------------------------------------------------------------------------
# tala keeper
# --------------------------------------------------------------------------
def test_a_lullaby_gets_three_and_the_tune_is_reset_to_it(lullaby):
    assert lullaby.beat.tala == "Tisra Eka"
    assert lullaby.beat.recast_tune
    assert lullaby.tune.melody.beats_per_cycle == 3
    assert any("re-set" in e.said for e in lullaby.events)


def test_with_no_pointer_in_the_mood_the_keeper_says_why_it_chose(lonely):
    assert lonely.beat.tala == "Adi"
    assert "bpm" in lonely.beat.reason
    assert lonely.beat.chosen_by == "tala keeper"


def test_the_beat_and_the_tune_share_one_cycle_and_one_tempo(lonely, lullaby):
    for draft in (lonely, lullaby):
        tala = tala_module.require(draft.beat.tala)
        assert draft.tune.melody.beats_per_cycle == tala.aksharas
        assert draft.beat.tempo_bpm == draft.tune.tempo_bpm
        assert draft.beat.beat.notes
        assert max(n.start for n in draft.beat.beat.notes) \
            < draft.tune.melody.duration


def test_the_producer_can_name_the_tala(team):
    draft = team.run(LONELY, "Tamil", tala="Misra Chapu", seed=2)
    assert draft.beat.tala == "Misra Chapu"
    assert draft.beat.chosen_by == "producer"
    assert draft.tune.melody.beats_per_cycle == 7
    with pytest.raises(KeyError):
        TalaKeeper().choose(draft.mood, draft.tune, draft.lyrics, "Waltz-ish")


# --------------------------------------------------------------------------
# the fit, and the hand-offs
# --------------------------------------------------------------------------
def test_every_phrase_is_sung_with_exactly_its_own_syllables(lonely, lullaby):
    for draft in (lonely, lullaby):
        fitted = draft.lyrics.fitted
        slots = build_slots(draft.tune.melody)
        assert fitted is not None and len(fitted.lines) == len(slots)
        for line, slot in zip(fitted.lines, slots):
            assert len(line.syllables) == slot.syllable_count, line.text
        assert not draft.lyrics.fit_notes
        sung = [n.syllable for n in draft.tune.melody.notes if n.syllable]
        assert len(sung) == len(draft.tune.melody.notes)


def test_a_rewrite_for_the_fit_is_in_the_log(lonely):
    for index in lonely.lyrics.revised:
        assert any(e.agent == LYRICIST and f"line {index + 1}" in e.said
                   for e in lonely.events)


def test_the_log_runs_in_krishs_order(lonely):
    agents = [e.agent for e in lonely.events]
    assert agents[0] == PRODUCER and agents[1] == MOOD_LISTENER
    assert set(agents[2:4]) == {LYRICIST, TUNESMITH}
    assert agents.index(TALA_KEEPER) > 3
    assert agents[-1] == PRODUCER
    assert lonely.events[1].handed_to == f"{LYRICIST} and {TUNESMITH}"


def test_the_draft_travels_as_json_and_says_what_is_not_done(lonely):
    data = json.loads(json.dumps(lonely.to_dict()))
    assert set(data) >= {"mood", "lyrics", "tune", "beat", "events",
                         "not_done"}
    assert data["tune"]["raaga"] == lonely.tune.raaga
    assert data["tune"]["alapana"] and data["tune"]["phrase"]
    assert data["lyrics"]["sung_phrases"]
    assert any("Prelude" in item for item in data["not_done"])


def test_the_same_seed_gives_the_same_song(team):
    first = team.run(LONELY, "Tamil", seed=21).to_dict()
    second = team.run(LONELY, "Tamil", seed=21).to_dict()
    assert first == second
