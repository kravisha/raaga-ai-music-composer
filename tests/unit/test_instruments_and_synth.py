"""Unit tests: the instrument catalog and the offline synthesis engine."""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.core.models import Note, Region, Track
from raagacomposer.music import instruments as catalog
from raagacomposer.music.synth import (render_note, render_notes,
                                       render_percussion, render_region,
                                       render_track)

pytestmark = pytest.mark.unit

SR = 22050


# --------------------------------------------------------------------------
# catalog identity
# --------------------------------------------------------------------------
def test_catalog_covers_the_instruments_the_spec_names():
    for key in ("veena", "violin", "saxophone", "piano", "mridangam", "guitar",
                "strings"):
        assert catalog.get(key) is not None, key


def test_every_entry_is_internally_consistent():
    for inst in catalog.all_instruments():
        assert inst.midi_low < inst.midi_high
        assert inst.roles
        assert inst.default_role in inst.roles
        assert inst.tags
        assert inst.description
        if inst.percussive:
            assert inst.hit_freqs and inst.hit_decays
        else:
            assert inst.harmonics


def test_exact_name_and_alias_lookup():
    assert catalog.find("Saxophone").key == "saxophone"
    assert catalog.find("sax").key == "saxophone"
    assert catalog.find("bansuri").key == "flute"
    assert catalog.find("tambura").key == "tanpura"
    assert catalog.find("drums").key == "drum_kit"


def test_unknown_instrument_returns_none_rather_than_guessing():
    assert catalog.find("theremin") is None
    assert catalog.find("hurdy gurdy") is None
    assert catalog.find_in_text("please add a theremin here") is None


def test_find_in_text_picks_the_longest_name():
    assert catalog.find_in_text("add the electric guitar here").key == "electric_guitar"
    assert catalog.find_in_text("use only piano for 15 seconds").key == "piano"


def test_closest_offers_alternatives_for_a_missing_instrument():
    names = [i.name for i in catalog.closest("violine")]
    assert "Violin" in names
    assert catalog.closest("theremin")          # always offers something


def test_feel_search_returns_instruments_that_match_the_words():
    ranked = catalog.suggest_for_feel(["lonely", "night", "warm"])
    assert ranked
    keys = [i.key for i, _ in ranked]
    assert any(k in ("saxophone", "cello", "piano", "electric_piano") for k in keys)


def test_feel_search_honours_avoid_and_role():
    ranked = catalog.suggest_for_feel(["lonely", "night"], avoid=["saxophone"])
    assert all(i.key != "saxophone" for i, _ in ranked)
    for inst, _ in catalog.suggest_for_feel(["cinematic", "grand"], role="pad"):
        assert inst.supports("pad")


def test_role_default_puts_percussion_on_rhythm():
    assert catalog.role_default(catalog.get("mridangam")) == "rhythm"
    assert catalog.role_default(catalog.get("veena"), "interlude") == "lead"


def test_describe_mentions_range_and_feel():
    text = catalog.describe(catalog.get("veena"))
    assert "Range" in text and "Feel" in text


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def _notes(n=6, start=0.0, step=0.5, midi=60):
    return [Note(swara="S", midi=midi + i, start=start + i * step, duration=0.4)
            for i in range(n)]


def test_a_single_note_renders_audible_finite_audio():
    inst = catalog.get("veena")
    audio = render_note(Note(midi=64, duration=0.6), inst, SR)
    assert len(audio) > SR * 0.5
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.01


def test_rendering_is_deterministic():
    inst = catalog.get("violin")
    a = render_notes(_notes(), inst, SR, total_seconds=4.0, seed=3)
    b = render_notes(_notes(), inst, SR, total_seconds=4.0, seed=3)
    assert np.array_equal(a, b)


def test_render_respects_the_requested_length():
    inst = catalog.get("piano")
    audio = render_notes(_notes(), inst, SR, total_seconds=5.0)
    assert len(audio) == int(5.0 * SR)


def test_no_notes_gives_silence_not_an_error():
    inst = catalog.get("piano")
    audio = render_notes([], inst, SR, total_seconds=2.0)
    assert len(audio) == int(2.0 * SR)
    assert not np.any(audio)


def test_notes_land_at_the_right_moment():
    inst = catalog.get("piano")
    late = [Note(midi=60, start=2.0, duration=0.5)]
    audio = render_notes(late, inst, SR, total_seconds=4.0)
    before = np.abs(audio[:int(1.9 * SR)]).max()
    after = np.abs(audio[int(2.0 * SR):int(2.4 * SR)]).max()
    assert before < 1e-6
    assert after > 0.01


def test_higher_velocity_is_louder():
    inst = catalog.get("piano")
    quiet = render_note(Note(midi=60, duration=0.5, velocity=40), inst, SR)
    loud = render_note(Note(midi=60, duration=0.5, velocity=110), inst, SR)
    assert np.abs(loud).max() > np.abs(quiet).max()


def test_gamaka_changes_the_signal():
    inst = catalog.get("veena")
    plain = render_note(Note(midi=62, duration=0.8), inst, SR, seed=1)
    bent = render_note(Note(midi=62, duration=0.8, gamaka="kampita"), inst, SR,
                       seed=1)
    assert len(plain) == len(bent)
    assert not np.allclose(plain, bent)


def test_percussion_renders_discrete_hits():
    inst = catalog.get("mridangam")
    hits = [Note(midi=36 + (i % 3), start=i * 0.5, duration=0.2) for i in range(6)]
    audio = render_percussion(hits, inst, SR, total_seconds=4.0)
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.01
    # There should be near-silence between strokes.
    assert np.abs(audio[int(0.4 * SR):int(0.48 * SR)]).max() < \
        np.abs(audio[:int(0.05 * SR)]).max()


def test_render_notes_routes_percussion_automatically():
    inst = catalog.get("tabla")
    audio = render_notes([Note(midi=36, start=0.0, duration=0.2)], inst, SR,
                         total_seconds=1.0)
    assert len(audio) == SR
    assert np.abs(audio).max() > 0.0


def test_region_and_track_rendering():
    region = Region(start=0.0, end=3.0, notes=_notes(4))
    audio = render_region(region, "veena", SR, total_seconds=3.0)
    assert len(audio) == int(3.0 * SR)
    track = Track(instrument="veena", regions=[region], gain=0.5)
    quiet = render_track(track, SR, 3.0)
    track.gain = 1.0
    loud = render_track(track, SR, 3.0)
    assert np.abs(loud).max() > np.abs(quiet).max()


def test_unknown_instrument_renders_silence_rather_than_crashing():
    region = Region(start=0.0, end=1.0, notes=_notes(2))
    audio = render_region(region, "no-such-instrument", SR, total_seconds=1.0)
    assert not np.any(audio)


def test_output_stays_within_a_sane_amplitude():
    inst = catalog.get("strings")
    audio = render_notes(_notes(12), inst, SR, total_seconds=8.0)
    assert float(np.abs(audio).max()) < 8.0
    assert np.isfinite(audio).all()
