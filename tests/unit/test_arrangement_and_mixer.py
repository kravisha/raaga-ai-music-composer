"""Unit tests: the orchestration engine, track model and mix engine."""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.core.models import CreativeBrief, Note, Region, Track
from raagacomposer.core.versioning import LockedContentError
from raagacomposer.music import arrangement as arranger
from raagacomposer.music import instruments as catalog
from raagacomposer.music import mixer
from raagacomposer.music.arrangement import PartRequest, describe, generate_part
from raagacomposer.music.melody import MelodyOptions, generate as gen_melody
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.unit

SR = 22050


@pytest.fixture(scope="module")
def melody():
    from raagacomposer.raaga.library import library
    raaga = library().require("Kambhoji")
    return gen_melody(raaga, MelodyOptions(tempo_bpm=76, seed=11,
                                           duration_target=120))


@pytest.fixture(scope="module")
def kambhoji():
    from raagacomposer.raaga.library import library
    return library().require("Kambhoji")


@pytest.fixture
def arrangement():
    return arranger.new_version(None)


# --------------------------------------------------------------------------
# part writing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["lead", "counter", "pad", "bass", "rhythm",
                                  "fill", "drone"])
def test_every_role_writes_notes_inside_its_window(melody, kambhoji, role):
    req = PartRequest(instrument="veena" if role != "rhythm" else "mridangam",
                      role=role, start=20.0, end=50.0, seed=3)
    notes = generate_part(melody, kambhoji, req)
    assert notes, role
    assert all(19.99 <= n.start <= 50.0 for n in notes), role
    assert all(n.duration > 0 for n in notes), role


def test_pitched_parts_stay_inside_the_raaga(melody, kambhoji):
    allowed = set(kambhoji.allowed)
    for role in ("lead", "counter", "pad", "bass", "drone", "fill"):
        req = PartRequest(instrument="veena", role=role, start=0.0, end=60.0,
                          seed=5)
        for note in generate_part(melody, kambhoji, req):
            assert parse_swara(note.swara)[0] in allowed, (role, note.swara)


def test_parts_stay_inside_the_instrument_range(melody, kambhoji):
    for key in ("veena", "flute", "bass", "cello"):
        inst = catalog.get(key)
        req = PartRequest(instrument=key, role=inst.default_role, start=0.0,
                          end=60.0, seed=2)
        for note in generate_part(melody, kambhoji, req):
            assert inst.midi_low <= note.midi <= inst.midi_high, key


def test_the_lead_part_follows_the_tune(melody, kambhoji):
    req = PartRequest(instrument="violin", role="lead", start=20.0, end=40.0)
    part = generate_part(melody, kambhoji, req)
    # A note already sounding at 20s is carried in and clipped to the window.
    tune = [n for n in melody.notes if n.start < 40.0 and n.end > 20.0]
    assert len(part) == len(tune)
    for a, b in zip(part, tune):
        assert a.start == pytest.approx(max(b.start, 20.0))
        assert (a.midi - b.midi) % 12 == 0        # same pitch class, any octave


def test_the_bass_sits_below_the_voice(melody, kambhoji):
    bass = generate_part(melody, kambhoji,
                         PartRequest(instrument="bass", role="bass",
                                     start=0.0, end=60.0))
    v_low, _ = arranger.vocal_register(melody)
    assert max(n.midi for n in bass) < v_low


def test_rhythm_parts_are_regular(melody, kambhoji):
    hits = generate_part(melody, kambhoji,
                         PartRequest(instrument="mridangam", role="rhythm",
                                     start=0.0, end=40.0, intensity=0.6))
    assert len(hits) > 10
    gaps = [b.start - a.start for a, b in zip(hits, hits[1:])]
    assert min(gaps) > 0


def test_density_changes_with_intensity(melody, kambhoji):
    sparse = generate_part(melody, kambhoji,
                           PartRequest(instrument="mridangam", role="rhythm",
                                       start=0.0, end=40.0, intensity=0.2))
    busy = generate_part(melody, kambhoji,
                         PartRequest(instrument="mridangam", role="rhythm",
                                     start=0.0, end=40.0, intensity=0.9))
    assert len(busy) > len(sparse)


def test_part_generation_is_deterministic(melody, kambhoji):
    req = PartRequest(instrument="veena", role="counter", start=10.0, end=40.0,
                      seed=17)
    a = generate_part(melody, kambhoji, req)
    b = generate_part(melody, kambhoji, req)
    assert [(n.start, n.midi) for n in a] == [(n.start, n.midi) for n in b]


def test_an_unknown_instrument_or_role_is_refused(melody, kambhoji):
    with pytest.raises(KeyError):
        generate_part(melody, kambhoji,
                      PartRequest(instrument="theremin", role="lead"))
    with pytest.raises(ValueError):
        generate_part(melody, kambhoji,
                      PartRequest(instrument="veena", role="nonsense"))


def test_register_choice_keeps_pads_under_the_voice(melody):
    inst = catalog.get("strings")
    low, high = arranger.choose_register(inst, "pad", melody, melody.tonic_midi)
    v_low, _ = arranger.vocal_register(melody)
    assert high <= v_low + 2


def test_role_suggestion_avoids_duplicating_what_is_playing(melody, arrangement,
                                                            kambhoji):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 60.0,
                            role="lead")
    role = arranger.suggest_role(catalog.get("violin"), arrangement)
    assert role != "lead"
    assert arranger.suggest_role(catalog.get("mridangam"), arrangement) == "rhythm"


# --------------------------------------------------------------------------
# arrangement operations
# --------------------------------------------------------------------------
def test_add_creates_a_track_and_region(melody, kambhoji, arrangement):
    track, region = arranger.add_instrument(arrangement, melody, kambhoji,
                                            "veena", 10.0, 30.0, role="lead")
    assert track in arrangement.tracks
    assert region in track.regions
    assert region.start == 10.0 and region.end == 30.0
    assert region.notes
    assert track.display_name == "Veena"


def test_adding_the_same_instrument_twice_extends_one_track(melody, kambhoji,
                                                            arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 20.0,
                            role="lead")
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 40.0, 60.0,
                            role="lead")
    tracks = arrangement.tracks_for_instrument("veena")
    assert len(tracks) == 1
    assert len(tracks[0].regions) == 2


def test_overlapping_add_replaces_the_unlocked_region(melody, kambhoji,
                                                      arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 40.0,
                            role="lead")
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 10.0, 20.0,
                            role="lead")
    regions = arrangement.tracks_for_instrument("veena")[0].regions
    assert len(regions) == 1
    assert (regions[0].start, regions[0].end) == (10.0, 20.0)


def test_remove_a_range_keeps_the_rest(melody, kambhoji, arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 20.0)
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 40.0, 60.0)
    removed = arranger.remove_instrument(arrangement, "veena", 0.0, 25.0)
    assert removed == 1
    remaining = arrangement.tracks_for_instrument("veena")[0].regions
    assert len(remaining) == 1 and remaining[0].start == 40.0


def test_remove_everything_drops_the_track(melody, kambhoji, arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 20.0)
    arranger.remove_instrument(arrangement, "veena")
    assert not arrangement.tracks_for_instrument("veena")


def test_replace_keeps_span_and_role(melody, kambhoji, arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "violin", 10.0, 30.0,
                            role="lead")
    arranger.replace_instrument(arrangement, melody, kambhoji, "violin",
                                "saxophone")
    assert not arrangement.tracks_for_instrument("violin")
    sax = arrangement.tracks_for_instrument("saxophone")[0]
    assert sax.role == "lead"
    assert (sax.regions[0].start, sax.regions[0].end) == (10.0, 30.0)
    assert sax.regions[0].notes


def test_replacing_an_instrument_that_is_not_there_is_reported(melody, kambhoji,
                                                               arrangement):
    with pytest.raises(LookupError):
        arranger.replace_instrument(arrangement, melody, kambhoji, "violin",
                                    "veena")


def test_replacing_with_an_unknown_instrument_is_refused(melody, kambhoji,
                                                         arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "violin", 0.0, 20.0)
    with pytest.raises(KeyError):
        arranger.replace_instrument(arrangement, melody, kambhoji, "violin",
                                    "theremin")


def test_regenerating_a_region_changes_only_that_region(melody, kambhoji,
                                                        arrangement):
    track, first = arranger.add_instrument(arrangement, melody, kambhoji,
                                           "veena", 0.0, 20.0, role="counter")
    _, second = arranger.add_instrument(arrangement, melody, kambhoji, "veena",
                                        40.0, 60.0, role="counter")
    untouched = [(n.start, n.midi) for n in second.notes]
    arranger.regenerate_region(arrangement, melody, kambhoji, track.id, first.id)
    assert [(n.start, n.midi) for n in second.notes] == untouched
    assert first.version == 2


def test_moving_a_region_regenerates_it_for_the_new_window(melody, kambhoji,
                                                           arrangement):
    track, region = arranger.add_instrument(arrangement, melody, kambhoji,
                                            "veena", 0.0, 20.0, role="lead")
    arranger.move_region(arrangement, melody, kambhoji, track.id, region.id,
                         40.0, 60.0)
    assert (region.start, region.end) == (40.0, 60.0)
    assert all(39.99 <= n.start <= 60.0 for n in region.notes)


# --------------------------------------------------------------------------
# locking
# --------------------------------------------------------------------------
def test_a_locked_region_refuses_an_overlapping_add(melody, kambhoji, arrangement):
    _, region = arranger.add_instrument(arrangement, melody, kambhoji, "veena",
                                        0.0, 40.0, role="lead")
    region.locked = True
    with pytest.raises(LockedContentError):
        arranger.add_instrument(arrangement, melody, kambhoji, "veena", 10.0,
                                20.0, role="lead")


def test_a_locked_track_refuses_changes(melody, kambhoji, arrangement):
    track, _ = arranger.add_instrument(arrangement, melody, kambhoji, "veena",
                                       0.0, 40.0)
    track.locked = True
    with pytest.raises(LockedContentError):
        arranger.remove_instrument(arrangement, "veena", 0.0, 40.0)
    with pytest.raises(LockedContentError):
        arranger.regenerate_region(arrangement, melody, kambhoji, track.id,
                                   track.regions[0].id)


def test_a_locked_region_survives_edits_elsewhere(melody, kambhoji, arrangement):
    track, locked = arranger.add_instrument(arrangement, melody, kambhoji,
                                            "veena", 0.0, 30.0, role="lead")
    locked.locked = True
    signature = [(n.start, n.midi) for n in locked.notes]
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 60.0, 90.0,
                            role="lead")
    arranger.add_instrument(arrangement, melody, kambhoji, "flute", 0.0, 90.0,
                            role="counter")
    assert [(n.start, n.midi) for n in locked.notes] == signature
    assert locked.locked


def test_lock_range_marks_every_overlapping_region(melody, kambhoji, arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 30.0)
    arranger.add_instrument(arrangement, melody, kambhoji, "flute", 0.0, 30.0,
                            role="counter")
    assert arranger.lock_range(arrangement, 10.0, 20.0, True) == 2
    assert all(r.locked for t in arrangement.tracks for r in t.regions)
    assert arranger.lock_range(arrangement, 10.0, 20.0, False) == 2


def test_set_region_lock_by_id(melody, kambhoji, arrangement):
    track, region = arranger.add_instrument(arrangement, melody, kambhoji,
                                            "veena", 0.0, 20.0)
    arranger.set_region_lock(arrangement, track.id, region.id, True)
    assert region.locked


# --------------------------------------------------------------------------
# versions and automatic arrangement
# --------------------------------------------------------------------------
def test_a_new_arrangement_version_copies_the_previous_tracks(melody, kambhoji,
                                                              arrangement):
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 20.0)
    second = arranger.new_version(arrangement)
    assert second.version == arrangement.version + 1
    assert len(second.tracks) == 1
    second.tracks[0].mute = True
    assert not arrangement.tracks[0].mute      # deep copy, not shared


def test_auto_arrange_builds_a_playable_bed(melody, kambhoji):
    brief = CreativeBrief(mood="longing", feel="lonely night", language="Tamil")
    built = arranger.auto_arrange(melody, kambhoji, brief, seed=4)
    assert len(built.tracks) >= 4
    roles = {t.role for t in built.tracks}
    assert {"drone", "pad", "bass", "rhythm"} <= roles
    assert all(t.regions for t in built.tracks)
    assert "Tanpura" in describe(built)


def test_auto_arrange_honours_preferred_and_avoided_instruments(melody, kambhoji):
    brief = CreativeBrief(mood="romantic", instruments_preferred=["flute"],
                          instruments_avoided=["saxophone"])
    built = arranger.auto_arrange(melody, kambhoji, brief, seed=6)
    keys = {t.instrument for t in built.tracks}
    assert "flute" in keys
    assert "saxophone" not in keys


def test_describe_reports_an_empty_arrangement():
    assert "no instruments" in describe(arranger.new_version(None))


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------
def _simple_arrangement(melody, kambhoji):
    arrangement = arranger.new_version(None)
    arranger.add_instrument(arrangement, melody, kambhoji, "veena", 0.0, 20.0,
                            role="lead")
    arranger.add_instrument(arrangement, melody, kambhoji, "mridangam", 0.0,
                            20.0, role="rhythm")
    return arrangement


def test_mix_produces_stereo_audio_at_the_requested_length(melody, kambhoji):
    result = mixer.mix(_simple_arrangement(melody, kambhoji), None, SR, 10.0,
                       kind="instrumental")
    assert result.audio.ndim == 2 and result.audio.shape[1] == 2
    assert result.duration == pytest.approx(10.0, abs=0.05)
    assert result.track_count == 2
    assert np.isfinite(result.audio).all()
    assert "track" in result.summary()


def test_mix_is_limited_and_normalised(melody, kambhoji):
    from raagacomposer.audio import dsp
    result = mixer.mix(_simple_arrangement(melody, kambhoji), None, SR, 10.0,
                       kind="instrumental")
    assert dsp.peak_db(result.audio) <= -0.7
    assert dsp.loudness_db(result.audio, SR) > -30


def test_muted_tracks_are_excluded(melody, kambhoji):
    arrangement = _simple_arrangement(melody, kambhoji)
    for track in arrangement.tracks:
        track.mute = True
    assert mixer.audible_tracks(arrangement) == []
    result = mixer.mix(arrangement, None, SR, 5.0, kind="instrumental")
    assert result.track_count == 0
    assert any("No audible" in note for note in result.notes)


def test_solo_wins_over_the_rest(melody, kambhoji):
    arrangement = _simple_arrangement(melody, kambhoji)
    arrangement.tracks[0].solo = True
    audible = mixer.audible_tracks(arrangement)
    assert len(audible) == 1 and audible[0] is arrangement.tracks[0]


def test_a_vocal_only_mix_ignores_the_instruments(melody, kambhoji):
    vocal = (np.sin(np.linspace(0, 400, SR * 5)) * 0.3).astype(np.float32)
    result = mixer.mix(_simple_arrangement(melody, kambhoji), vocal, SR, 5.0,
                       kind="vocal_only")
    assert result.track_count == 0
    assert np.abs(result.audio).max() > 0.0


def test_a_full_mix_without_a_vocal_says_so(melody, kambhoji):
    result = mixer.mix(_simple_arrangement(melody, kambhoji), None, SR, 5.0,
                       kind="full")
    assert any("No vocal" in note for note in result.notes)


def test_stems_are_produced_per_track(melody, kambhoji):
    stems = mixer.stems(_simple_arrangement(melody, kambhoji), SR, 10.0)
    assert len(stems) == 2
    for name, audio in stems.items():
        assert audio.ndim == 2
        assert np.isfinite(audio).all()


def test_mix_reports_progress_and_honours_cancellation(melody, kambhoji):
    seen = []
    mixer.mix(_simple_arrangement(melody, kambhoji), None, SR, 5.0,
              kind="instrumental", progress=lambda p, m: seen.append(p))
    assert seen and max(seen) <= 1.0
    stopped = mixer.mix(_simple_arrangement(melody, kambhoji), None, SR, 5.0,
                        kind="instrumental", cancelled=lambda: True)
    assert stopped.audio is not None


def test_the_bass_stays_below_a_voice_that_reaches_the_bass_register(kambhoji):
    """The bass register was tonic-24..tonic-5 whatever the voice did; a tune
    whose lowest sung note is tonic-5 met the bass on that note (the planner
    change of 2026-09-08 gave the Kambhoji fixture such a tune)."""
    from raagacomposer.core.models import MelodyVersion, Section, SectionKind
    tonic = 60
    melody = MelodyVersion(version=1, raaga="Kambhoji", tonic_midi=tonic, tempo_bpm=76,
                           sections=[Section(name="Pallavi", kind=SectionKind.PALLAVI,
                                             start=0.0, end=60.0)])
    section = melody.sections[0]
    t = 0.0
    for midi, swara in ((tonic - 5, "P-"), (tonic - 3, "D2-"), (tonic, "S"),
                        (tonic + 4, "G3"), (tonic - 5, "P-"), (tonic + 2, "R2")):
        melody.notes.append(Note(swara=swara, midi=midi, start=t, duration=2.0,
                                 velocity=80, section_id=section.id))
        t += 2.0
    v_low, _ = arranger.vocal_register(melody)
    assert v_low == tonic - 5
    inst = catalog.get("bass")
    low, high = arranger.choose_register(inst, "bass", melody, tonic)
    assert high < v_low, (low, high)
    bass = generate_part(melody, kambhoji,
                         PartRequest(instrument="bass", role="bass", start=0.0, end=12.0))
    assert bass and max(n.midi for n in bass) < v_low


# --------------------------------------------------------------------------
# a locked region survives the next auto arrangement
# --------------------------------------------------------------------------
def _regenerated(melody, kambhoji, name="Charanam 1", seed=12):
    from raagacomposer.music.melody import regenerate_section
    section = next(s for s in melody.sections if s.name == name)
    return regenerate_section(melody, kambhoji, section.id,
                              MelodyOptions(tempo_bpm=76, seed=seed,
                                            duration_target=120), 2)


def test_auto_arrange_keeps_a_locked_region_and_rebuilds_around_it(melody, kambhoji):
    """One locked region used to make every later auto-arrangement raise
    ("has locked regions at 0-120s. Unlock them first."): the carried
    region sat inside the whole-song span the pad was added over."""
    brief = CreativeBrief(language="Tamil")
    first = arranger.auto_arrange(melody, kambhoji, brief, seed=5)
    lead = next(t for t in first.tracks if t.role == "lead")
    assert len(lead.regions) >= 2, "the lead plays more than one instrumental section"
    interlude = next(s for s in melody.sections if s.name == "Interlude 1")
    region = next(r for r in lead.regions if r.overlaps(interlude.start, interlude.end))
    arranger.set_region_lock(first, lead.id, region.id, True)
    kept = (region.start, region.end, region.seed, [(n.start, n.midi) for n in region.notes])
    pad_before = [(r.start, r.end) for r in next(t for t in first.tracks if t.role == "pad").regions]

    rewritten = _regenerated(melody, kambhoji)
    second = arranger.auto_arrange(rewritten, kambhoji, brief, seed=5, previous=first)
    lead2 = next(t for t in second.tracks if t.role == "lead")
    still = [(r.start, r.end, r.seed, [(n.start, n.midi) for n in r.notes])
             for r in lead2.regions if r.locked]
    assert still == [kept], "the locked region did not come through untouched"
    # The lead's other sections are rebuilt around it, and nowhere over it.
    unlocked = [r for r in lead2.regions if not r.locked]
    assert unlocked and all(not r.overlaps(kept[0], kept[1]) for r in unlocked)
    assert len(lead2.regions) == len(lead.regions)
    # And the other tracks are rebuilt as before: nothing raised, every
    # role present once, the pad over the whole new tune.
    roles = [t.role for t in second.tracks]
    assert roles.count("pad") == 1 and roles.count("bass") == 1 and roles.count("lead") == 1
    pad2 = next(t for t in second.tracks if t.role == "pad")
    assert [(r.start, r.end) for r in pad2.regions] == pad_before
    assert not any(r.locked for r in pad2.regions)


def _pallavi_of_two_notes():
    """Arya's fixture: an 8-second Pallavi at 120 bpm in Hamsadhwani, two
    sustained notes, 0-4 and 4-8 s."""
    from raagacomposer.core.models import MelodyVersion, Section, SectionKind
    from raagacomposer.raaga.library import library
    section = Section(name="Pallavi", kind=SectionKind.PALLAVI, start=0.0, end=8.0)
    melody = MelodyVersion(raaga="Hamsadhwani", tonic_midi=60, tempo_bpm=120,
                           sections=[section],
                           notes=[Note(swara="S", midi=60, start=0.0, duration=4.0,
                                       section_id=section.id),
                                  Note(swara="P", midi=67, start=4.0, duration=4.0,
                                       section_id=section.id)])
    return melody, library().require("Hamsadhwani")


def _locked_rest(instrument, role, lo, hi):
    from raagacomposer.core.models import Region, Track
    return Track(instrument=instrument, role=role,
                 regions=[Region(start=lo, end=hi, role=role, locked=True, notes=[])])


def _generated_notes(arrangement, instrument, role):
    track = next(t for t in arrangement.tracks
                 if t.instrument == instrument and t.role == role)
    return [(r, n) for r in track.regions if not r.locked for n in r.notes]


def _dry(arrangement, instrument, role, sr=8000, seconds=8.0):
    """The generated regions of one track, rendered alone."""
    from raagacomposer.core.models import Track
    from raagacomposer.music.synth import render_track
    track = next(t for t in arrangement.tracks
                 if t.instrument == instrument and t.role == role)
    only = Track(instrument=track.instrument, role=track.role,
                 regions=[r for r in track.regions if not r.locked])
    return render_track(only, sr, seconds), sr


def _rms(audio, sr, lo, hi):
    seg = audio[int(lo * sr):int(hi * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def test_a_counter_line_writes_nothing_inside_a_locked_rest(arrangement):
    """Arya's P1: the counter took a window from the melody note that began
    at 0 s, so the 2-8 s region held notes at 0.5, 1.25 and 1.5 s, inside
    the locked 0-2 s rest, though the rectangles did not overlap."""
    melody, hamsadhwani = _pallavi_of_two_notes()
    previous = arranger.new_version(None)
    previous.tracks.append(_locked_rest("veena", "counter", 0.0, 2.0))
    brief = CreativeBrief(language="Tamil", instruments_preferred=["veena"])
    out = arranger.auto_arrange(melody, hamsadhwani, brief, seed=5, previous=previous,
                                lead=catalog.get("veena"))
    generated = _generated_notes(out, "veena", "counter")
    assert generated, "the counter wrote nothing at all"
    for region, note in generated:
        assert note.start >= region.start - 1e-6 and note.start + note.duration <= region.end + 1e-6, \
            (region.start, region.end, note.start, note.duration)
        assert note.start >= 2.0, note.start
    # The kept rest is untouched, and the lock is byte-equal.
    kept = next(r for t in out.tracks if t.instrument == "veena" for r in t.regions if r.locked)
    assert (kept.start, kept.end, kept.notes) == (0.0, 2.0, [])
    # Render path: the generated regions alone put nothing in the rest's
    # interior (a tail from a note ending at a boundary is allowed room).
    audio, sr = _dry(out, "veena", "counter")
    assert _rms(audio, sr, 0.3, 1.95) < 1e-4, _rms(audio, sr, 0.3, 1.95)
    assert _rms(audio, sr, 2.5, 7.5) > 1e-3, "the part still plays where it may"


def test_a_rhythm_stroke_stops_at_a_locked_rest(arrangement):
    """Arya's P2: the stroke at 2.0 s lasted 0.2 s into the 2.1-3.1 s lock."""
    melody, hamsadhwani = _pallavi_of_two_notes()
    previous = arranger.new_version(None)
    previous.tracks.append(_locked_rest("mridangam", "rhythm", 2.1, 3.1))
    brief = CreativeBrief(language="Tamil", instruments_preferred=["mridangam"])
    out = arranger.auto_arrange(melody, hamsadhwani, brief, seed=5, previous=previous,
                                lead=catalog.get("veena"))
    generated = _generated_notes(out, "mridangam", "rhythm")
    assert generated
    for region, note in generated:
        assert note.start + note.duration <= region.end + 1e-6, (region.end, note.start, note.duration)
        assert note.start >= region.start - 1e-6
        assert not (note.start < 3.1 and note.start + note.duration > 2.1), (note.start, note.duration)
    assert any(abs(n.start - 2.0) < 1e-6 for _, n in generated), "the stroke at 2.0 s is kept, cut short"
    # Render path: a stroke's sample rings past its event, and that tail is
    # allowed; what must not appear is a new stroke inside the rest.  So
    # the rest's interior is quieter than the stroke before it and has no
    # onset louder than that stroke's own decay.
    audio, sr = _dry(out, "mridangam", "rhythm")
    stroke = _rms(audio, sr, 2.0, 2.2)
    assert stroke > 1e-3, "the stroke at 2.0 s is silent"
    assert _rms(audio, sr, 2.5, 3.05) < stroke * 0.5, (stroke, _rms(audio, sr, 2.5, 3.05))
    peak_in_rest = float(np.max(np.abs(audio[int(2.6 * sr):int(3.05 * sr)])))
    peak_of_stroke = float(np.max(np.abs(audio[int(2.0 * sr):int(2.3 * sr)])))
    assert peak_in_rest < peak_of_stroke * 0.5, (peak_in_rest, peak_of_stroke)


def test_every_part_writer_stays_inside_its_span(kambhoji, melody):
    """A part is the region's to write and nowhere else: whatever the
    writer produced, nothing starts before the span or ends after it."""
    for role in ("lead", "counter", "pad", "bass", "rhythm", "fill", "drone"):
        for start, end in ((13.0, 27.5), (0.0, 2.1), (2.0, 8.0)):
            notes = generate_part(melody, kambhoji,
                                  PartRequest(instrument="mridangam" if role == "rhythm" else "veena",
                                              role=role, start=start, end=end, seed=3))
            for n in notes:
                assert n.start >= start - 1e-6, (role, start, end, n.start)
                assert n.start + n.duration <= end + 1e-6, (role, start, end, n.start, n.duration)
                assert n.duration >= 0.03


def test_auto_arrange_leaves_a_locked_track_alone(melody, kambhoji):
    brief = CreativeBrief(language="Tamil")
    first = arranger.auto_arrange(melody, kambhoji, brief, seed=5)
    bass = next(t for t in first.tracks if t.role == "bass")
    bass.locked = True
    signature = [(r.start, r.end, [(n.start, n.midi) for n in r.notes]) for r in bass.regions]
    second = arranger.auto_arrange(_regenerated(melody, kambhoji), kambhoji, brief,
                                   seed=5, previous=first)
    bass2 = next(t for t in second.tracks if t.role == "bass")
    assert bass2.locked
    assert [(r.start, r.end, [(n.start, n.midi) for n in r.notes]) for r in bass2.regions] \
        == signature
