"""Independent checks of the requested add-tala workflow, with silent audio stubs."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.music import tala


@pytest.fixture
def composer(app, monkeypatch):
    # Exercise real composition, percussion and arrangement; never play audio.
    real_render = app.render
    monkeypatch.setattr(app, "render", lambda *a, **k: None)
    monkeypatch.setattr(app, "render_beat", lambda *a, **k: None)
    app.new_project("Independent tala review", write=False)
    app.update_brief(language="Tamil", duration_target=20, mood="", feel="",
                     situation="", notes="", tala="", tempo_preference=108)
    app.select_raaga("Hamsadhwani")
    return app, real_render


@pytest.mark.parametrize("situation,cycle", [
    ("a chase through a city", 5), ("a lullaby for a child", 3),
    ("a village folk festival", 7),
])
def test_automatic_choice_drives_composition(composer, settle, situation, cycle):
    app, _ = composer
    app.update_brief(situation=situation)
    choice = app.tala_choice()
    app.generate_tune(seed=24)
    settle()
    assert choice.reason and not choice.chosen_by_creator
    assert app.project.melody().beats_per_cycle == cycle
    assert app.project.melody().tempo_bpm == 108


@pytest.mark.parametrize("name,cycle", [("Rupaka", 6), ("Adi", 8)])
def test_explicit_cycle_wins_over_conflicting_brief(composer, settle, name, cycle):
    app, _ = composer
    app.update_brief(situation="an urgent chase", tala=name)
    assert app.tala_choice().chosen_by_creator
    app.generate_tune(seed=24)
    settle()
    assert app.project.melody().beats_per_cycle == cycle


def prepare_tune(app, settle):
    app.update_brief(situation="a lullaby for a child")
    app.generate_tune(seed=24)
    settle()
    assert app.project.melody() is not None


def test_adding_percussion_preserves_all_melody_fields_and_locks(composer, settle):
    app, _ = composer
    prepare_tune(app, settle)
    melody = app.project.melody()
    for section in melody.sections:
        section.locked = True
    before = deepcopy(asdict(melody))
    app.update_brief(situation="a fast urgent chase")
    app.generate_beat("busy", seed=21)
    settle()
    beat = app.project.beat()
    assert beat and beat.notes
    assert tala.require(beat.tala).aksharas == melody.beats_per_cycle == 3
    assert beat.tempo_bpm == melody.tempo_bpm
    assert beat.duration == melody.duration
    assert all(0 <= n.start < melody.duration for n in beat.notes)
    assert asdict(app.project.melody()) == before


def capture_mix(monkeypatch):
    from raagacomposer import app as app_module
    calls = []
    def fake_mix(arrangement, vocal_audio, sr, total, **kwargs):
        calls.append(deepcopy(arrangement))
        return SimpleNamespace(audio=np.zeros((32, 2), dtype=np.float32),
                               notes=[], loudness_db=-20, track_count=0,
                               summary=lambda: "silent independent mix inspection")
    monkeypatch.setattr(app_module.mixer, "mix", fake_mix)
    return calls


def assert_current_beat_in_mix(app, arrangement):
    beat = app.project.beat()
    assert arrangement is not None, "Add beat followed by full mix has no arrangement or percussion"
    regions = [r for t in arrangement.tracks if t.role == "rhythm" for r in t.regions]
    versions = {r.meta.get("beat_version") for r in regions}
    assert str(beat.version) in versions, f"Selected beat v{beat.version}; full mix still uses {versions}"
    first_sung = next((s.start for s in app.project.melody().sections if not s.kind.instrumental), 0)
    expected = [asdict(n) for n in beat.notes if n.start >= first_sung]
    actual = [asdict(n) for r in regions if r.meta.get("beat_version") == str(beat.version) for n in r.notes]
    assert actual == expected, "The selected strokes were changed before entering the mix"


def test_explicit_arrangement_includes_selected_beat(composer, settle, monkeypatch):
    app, real_render = composer
    prepare_tune(app, settle)
    app.generate_beat("busy", seed=21)
    settle()
    app.auto_arrange()
    settle()
    calls = capture_mix(monkeypatch)
    real_render("full", autoplay=False)
    settle()
    assert calls
    assert_current_beat_in_mix(app, calls[-1])


def test_add_beat_reaches_full_mix_without_extra_arrange_action(composer, settle, monkeypatch):
    app, real_render = composer
    prepare_tune(app, settle)
    app.generate_beat("busy", seed=21)
    settle()
    calls = capture_mix(monkeypatch)
    real_render("full", autoplay=False)
    settle()
    assert calls
    assert_current_beat_in_mix(app, calls[-1])


def test_revised_beat_reaches_existing_mix_and_keeps_other_tracks(composer, settle, monkeypatch):
    app, real_render = composer
    prepare_tune(app, settle)
    app.generate_beat("sparse", seed=21)
    settle()
    app.auto_arrange()
    settle()
    arrangement = app.project.arrangement()
    others = [t for t in arrangement.tracks if t.role != "rhythm"]
    assert others
    others[0].locked = True
    before = [asdict(t) for t in others]
    app.generate_beat("busy", seed=22)
    settle()
    assert [asdict(t) for t in app.project.arrangement().tracks if t.role != "rhythm"] == before
    calls = capture_mix(monkeypatch)
    real_render("full", autoplay=False)
    settle()
    assert calls
    assert_current_beat_in_mix(app, calls[-1])


def test_purchase_is_not_a_chase_request():
    choice = tala.suggest(CreativeBrief(situation="a quiet purchase of a gift", mood="", feel=""))
    assert '"chase" in' not in choice.reason, choice.describe()


def test_rejected_chase_is_not_the_reason_for_a_lullaby():
    choice = tala.suggest(CreativeBrief(situation="a lullaby, not a chase", mood="", feel=""))
    assert '"chase" in' not in choice.reason, choice.describe()


def test_auto_arrange_keeps_one_beat_layer(composer, settle):
    app, _ = composer
    prepare_tune(app, settle)
    app.generate_beat("sparse", seed=21)
    settle()
    app.auto_arrange()
    settle()
    rhythms = [t for t in app.project.arrangement().tracks if t.role == "rhythm"]
    assert len(rhythms) == 1, "Auto Arrange doubled the existing rhythm layer"


def test_locked_rhythm_track_is_not_replaced(composer, settle):
    app, _ = composer
    prepare_tune(app, settle)
    app.generate_beat("sparse", seed=21)
    settle()
    track = next(t for t in app.project.arrangement().tracks if t.role == "rhythm")
    track.locked = True
    before = deepcopy(asdict(track))
    app.generate_beat("busy", seed=22)
    settle()
    assert asdict(app.project.arrangement().track_by_id(track.id)) == before


def test_locked_rhythm_region_is_not_replaced(composer, settle):
    app, _ = composer
    prepare_tune(app, settle)
    app.generate_beat("sparse", seed=21)
    settle()
    track = next(t for t in app.project.arrangement().tracks if t.role == "rhythm")
    track.regions[0].locked = True
    assert not track.locked, "the creator locked one region, not the whole track"
    before = deepcopy(asdict(track.regions[0]))
    app.generate_beat("busy", seed=22)
    settle()
    regions = app.project.arrangement().track_by_id(track.id).regions
    assert before in [asdict(r) for r in regions], "Adding a beat overwrote the creator's locked percussion region"


def test_one_undo_keeps_beat_selection_and_mix_in_agreement(composer, settle):
    app, _ = composer
    prepare_tune(app, settle)
    app.generate_beat("sparse", seed=21)
    settle()
    first_version = app.project.beat().version
    app.generate_beat("busy", seed=22)
    settle()
    assert app.project.beat().version != first_version
    assert app.undo_action()
    settle()
    assert_current_beat_in_mix(app, app.project.arrangement())
    assert app.project.beat().version == first_version, "One undo did not revert the whole Add Beat action"


def test_explicit_lullaby_situation_beats_the_default_mood():
    choice = tala.suggest(CreativeBrief(situation="a lullaby for a child"))
    assert choice.tala.name == "Tisra Eka", choice.describe()
