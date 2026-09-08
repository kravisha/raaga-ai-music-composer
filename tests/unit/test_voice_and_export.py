"""Unit tests: voice profiles, singing synthesis, providers and export."""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.audio import export as export_engine
from raagacomposer.core.models import (CreativeBrief, LyricsVersion,
                                       VocalDirection, VoiceProfile)
from raagacomposer.lyrics.generator import generate as gen_lyrics
from raagacomposer.music.melody import MelodyOptions, generate as gen_melody
from raagacomposer.providers import registry
from raagacomposer.providers.local import (LocalLLM, LocalMusicProvider,
                                           LocalVoiceProvider)
from raagacomposer.voice.profiles import BUILTIN, VoiceProfileManager
from raagacomposer.voice.renderer import (STYLE_PRESETS, plan_segments, render,
                                          render_melody, split_syllable)

pytestmark = pytest.mark.unit

SR = 22050


@pytest.fixture(scope="module")
def melody():
    from raagacomposer.raaga.library import library
    return gen_melody(library().require("Charukesi"),
                      MelodyOptions(tempo_bpm=70, seed=13, duration_target=90))


@pytest.fixture(scope="module")
def lyrics(melody):
    return gen_lyrics(melody, CreativeBrief(language="Tamil"), seed=5)


# --------------------------------------------------------------------------
# voice profiles
# --------------------------------------------------------------------------
def test_builtin_profiles_are_sane():
    for profile in BUILTIN:
        assert profile.range_low < profile.base_midi < profile.range_high
        assert profile.builtin
        assert 0.0 <= profile.breathiness <= 1.0


def test_profile_lookup(tmp_path):
    manager = VoiceProfileManager(tmp_path / "voices.json")
    assert manager.get(BUILTIN[0].id) is BUILTIN[0]
    assert manager.by_name("Female - Warm").gender == "female"
    assert manager.by_name("male").gender == "male"
    assert manager.default("male").gender == "male"
    assert manager.get("nope") is None


def test_user_profiles_persist(tmp_path):
    path = tmp_path / "voices.json"
    manager = VoiceProfileManager(path)
    added = manager.add(VoiceProfile(id="voice_test", name="Mine",
                                     gender="female", builtin=False))
    assert added in manager.all()
    reloaded = VoiceProfileManager(path)
    assert reloaded.get("voice_test") is not None
    assert reloaded.remove("voice_test")
    assert VoiceProfileManager(path).get("voice_test") is None


def test_builtin_profiles_cannot_be_edited(tmp_path):
    manager = VoiceProfileManager(tmp_path / "voices.json")
    with pytest.raises(ValueError):
        manager.update(BUILTIN[0])
    clone = manager.duplicate(BUILTIN[0].id, "My Copy")
    assert not clone.builtin and clone.name == "My Copy"


def test_a_profile_can_be_derived_from_a_recording(tmp_path):
    # A synthetic 165 Hz tone stands in for a supplied vocal recording.
    t = np.arange(int(SR * 1.5)) / SR
    tone = (0.4 * np.sin(2 * np.pi * 165 * t)
            + 0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    wav = tmp_path / "sample.wav"
    sf.write(str(wav), tone, SR)

    manager = VoiceProfileManager(tmp_path / "voices.json")
    profile = manager.create_from_recording([str(wav)], "From Recording")
    assert not profile.builtin
    assert profile.source_samples == [str(wav)]
    assert 45 <= profile.base_midi <= 70          # around E3
    assert profile.range_low < profile.base_midi < profile.range_high
    assert "median pitch" in profile.notes


def test_a_silent_recording_is_rejected_clearly(tmp_path):
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(SR, dtype=np.float32), SR)
    manager = VoiceProfileManager(tmp_path / "voices.json")
    with pytest.raises(ValueError):
        manager.create_from_recording([str(wav)], "Silence")


# --------------------------------------------------------------------------
# singing renderer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("syllable,consonant,vowel", [
    ("ka", "k", "a"),
    ("nee", "n", "ee"),
    ("iravu", "", "i"),
    ("sha", "sh", "a"),
    ("", "", "a"),
])
def test_syllables_split_into_consonant_and_vowel(syllable, consonant, vowel):
    assert split_syllable(syllable)[:2] == (consonant, vowel)


@pytest.mark.parametrize("syllable,coda", [
    ("vaan", "n"),
    ("kal", "l"),
    ("nee", ""),
    ("thaayk", "yk"),
    ("iravu", ""),
    ("", ""),
])
def test_a_syllable_keeps_the_consonant_that_closes_it(syllable, coda):
    """The coda used to be discarded, so "vaan" was sung "vaa".

    A large part of what makes a word recognisable is at its end.
    """
    assert split_syllable(syllable)[2] == coda


def test_a_second_vowel_is_not_taken_as_a_coda():
    """"iravu" is more than one syllable written into one slot.  Guessing
    where to split it would put sounds on notes nobody wrote them for."""
    onset, vowel, coda = split_syllable("iravu")
    assert (onset, vowel) == ("", "i")
    assert coda == ""


def test_segments_skip_the_instrumental_sections(melody, lyrics):
    segments = plan_segments(melody, lyrics)
    assert segments
    instrumental = {s.id for s in melody.sections if s.kind.instrumental}
    for segment in segments:
        note = next(n for n in melody.notes
                    if n.start == pytest.approx(segment.start))
        assert note.section_id not in instrumental


def test_segments_carry_the_fitted_syllables(melody, lyrics):
    segments = plan_segments(melody, lyrics)
    assert any(s.syllable for s in segments)


def test_a_vocal_render_is_finite_audio_of_the_right_length(melody, lyrics):
    profile = BUILTIN[0]
    audio = render_melody(melody, lyrics, profile, VocalDirection(), SR,
                          total_seconds=melody.duration + 1.0)
    assert len(audio) == int((melody.duration + 1.0) * SR)
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.1


def test_rendering_is_deterministic(melody, lyrics):
    a = render_melody(melody, lyrics, BUILTIN[0], VocalDirection(), SR, seed=4)
    b = render_melody(melody, lyrics, BUILTIN[0], VocalDirection(), SR, seed=4)
    assert np.array_equal(a, b)


def test_changing_the_singer_changes_the_timbre_not_the_notes(melody, lyrics):
    female = render_melody(melody, lyrics, BUILTIN[0], VocalDirection(), SR,
                           seed=2)
    male = render_melody(melody, lyrics, BUILTIN[2], VocalDirection(), SR, seed=2)
    assert len(female) == len(male)
    assert not np.allclose(female, male)


def test_vocal_direction_changes_the_delivery(melody, lyrics):
    soft = render_melody(melody, lyrics, BUILTIN[0],
                         VocalDirection(style="soft", intensity=0.2), SR, seed=1)
    strong = render_melody(melody, lyrics, BUILTIN[0],
                           VocalDirection(style="strong", intensity=1.0), SR,
                           seed=1)
    assert not np.allclose(soft, strong)


def test_every_named_style_renders(melody, lyrics):
    for style in STYLE_PRESETS:
        audio = render_melody(melody, lyrics, BUILTIN[0],
                              VocalDirection(style=style), SR, seed=1)
        assert np.isfinite(audio).all(), style


def test_a_tune_with_no_lyrics_still_sings(melody):
    audio = render_melody(melody, None, BUILTIN[0], VocalDirection(), SR, seed=1)
    assert float(np.abs(audio).max()) > 0.05


def test_no_segments_gives_silence():
    audio = render([], BUILTIN[0], VocalDirection(), SR, total_seconds=2.0)
    assert len(audio) == int(2.0 * SR)
    assert not np.any(audio)


def test_rests_are_quieter_than_sung_notes(melody, lyrics):
    segments = plan_segments(melody, lyrics)
    audio = render(segments, BUILTIN[0], VocalDirection(style="strong"), SR,
                   total_seconds=melody.duration + 1.0, seed=3)
    first = segments[0]
    sung = np.abs(audio[int(first.start * SR):int(first.end * SR)]).max()
    silent_tail = np.abs(audio[int((melody.duration + 0.5) * SR):]).max()
    assert sung > silent_tail


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
def test_local_providers_are_always_available():
    music = LocalMusicProvider()
    voice = LocalVoiceProvider()
    assert music.available and voice.available
    assert len(music.instruments()) > 20
    assert voice.voices()
    assert music.info().kind == "music"


def test_local_music_provider_rejects_an_unknown_instrument():
    with pytest.raises(KeyError):
        LocalMusicProvider().render_part([], "theremin", SR)


def test_the_local_llm_reports_itself_unavailable():
    llm = LocalLLM()
    assert not llm.available
    assert llm.write_lyrics([], CreativeBrief()) == []
    assert llm.classify_intent("x", []) == {}
    assert "not configured" in llm.status()


def test_the_registry_always_returns_a_working_set(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings.llm_provider = "local"
    providers = registry.build(settings, stt_name="typed")
    assert providers.music.available and providers.voice.available
    assert providers.notes                      # explains the local fallback
    assert "local-synth" in providers.summary()


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def test_wav_export_round_trips(tmp_path):
    audio = (np.sin(np.linspace(0, 200, SR)) * 0.4).astype(np.float32)
    path = export_engine.write_wav(tmp_path / "out.wav", audio, SR)
    assert path.exists()
    data, sr = sf.read(str(path), always_2d=True)
    assert sr == SR
    assert data.shape[1] == 2                  # mono is written as stereo
    assert abs(len(data) - len(audio)) <= 1


def test_export_audio_picks_the_format(tmp_path):
    audio = np.zeros(SR, dtype=np.float32)
    assert export_engine.export_audio(tmp_path / "a.wav", audio, SR).suffix == ".wav"
    if export_engine.ffmpeg_path() is None:
        with pytest.raises(RuntimeError, match="ffmpeg"):
            export_engine.export_audio(tmp_path / "a.mp3", audio, SR)


def test_stems_are_written_with_safe_names(tmp_path):
    stems = {"Veena (lead)": np.zeros(SR, np.float32),
             "Mridangam/rhythm": np.zeros(SR, np.float32)}
    paths = export_engine.export_stems(tmp_path / "stems", stems, SR)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_midi_export_is_a_valid_file(tmp_path, melody):
    path = export_engine.write_midi(tmp_path / "tune.mid", melody)
    raw = path.read_bytes()
    assert raw[:4] == b"MThd"
    length, fmt, tracks, division = struct.unpack(">IHHH", raw[4:14])
    assert length == 6 and fmt == 1 and tracks >= 2 and division == 480
    assert raw.count(b"MTrk") == tracks


def test_midi_export_includes_the_arrangement(tmp_path, melody):
    from raagacomposer.music import arrangement as arranger
    from raagacomposer.raaga.library import library
    raaga = library().require("Charukesi")
    built = arranger.new_version(None)
    arranger.add_instrument(built, melody, raaga, "veena", 0.0, 30.0)
    path = export_engine.write_midi(tmp_path / "full.mid", melody, built)
    raw = path.read_bytes()
    assert raw.count(b"MTrk") >= 3
    assert b"Veena" in raw


def test_musicxml_is_well_formed(tmp_path, melody, lyrics):
    import xml.etree.ElementTree as ET
    path = export_engine.write_musicxml(tmp_path / "tune.musicxml", melody, lyrics)
    tree = ET.parse(path)
    root = tree.getroot()
    assert root.tag == "score-partwise"
    assert root.findall(".//measure")
    assert root.findall(".//note/pitch/step")
    assert root.findall(".//lyric/text")


def test_lyrics_text_export_groups_by_section(tmp_path, melody, lyrics):
    path = export_engine.write_lyrics_text(tmp_path / "lyrics.txt", lyrics, melody)
    text = path.read_text(encoding="utf-8")
    assert "[" in text and "]" in text
    assert lyrics.lines[0].text in text


def test_project_archive_contains_the_files(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / "mixes").mkdir(parents=True)
    (project_dir / "project.json").write_text("{}", encoding="utf-8")
    (project_dir / "mixes" / "full.wav").write_bytes(b"RIFF")
    (project_dir / "scratch.tmp").write_text("skip me", encoding="utf-8")

    archive = export_engine.archive_project(tmp_path / "out.zip", project_dir)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
    assert "project.json" in names
    assert "mixes/full.wav" in names
    assert not any(n.endswith(".tmp") for n in names)


# --------------------------------------------------------------------------
# a hum is a closed mouth, not an open vowel
# --------------------------------------------------------------------------
def test_a_hum_is_darker_than_a_sung_vowel():
    """Reported: the hummed tune still sounded like an instrument.

    It was genuinely going through the voice renderer - but an open "aa"
    through four strong formants is close to how you would synthesise a
    reed, so it read as one.  A closed hum has a low first resonance and
    almost nothing above it, and that is what the ear uses to tell a shut
    mouth from an open one.
    """
    import numpy as np

    from raagacomposer.core.models import VocalDirection
    from raagacomposer.voice import renderer
    from raagacomposer.voice.profiles import BUILTIN

    profile = BUILTIN[0]
    sr = 22050
    notes = [renderer.SungSegment(start=i * 0.5, end=i * 0.5 + 0.45,
                                  midi=60 + i, vowel="a")
             for i in range(6)]
    open_vowel = renderer.render(notes, profile, VocalDirection(), sr, 3.5, seed=5)
    for seg in notes:
        seg.vowel = "hum"
    hummed = renderer.render(notes, profile, VocalDirection(), sr, 3.5, seed=5)

    def above(x, hz):
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / sr)
        return float(spectrum[freqs > hz].sum() / max(spectrum.sum(), 1e-9))

    assert above(hummed, 1000) < above(open_vowel, 1000) / 5, (
        f"the hum is not appreciably darker: "
        f"{above(hummed, 1000):.4f} vs {above(open_vowel, 1000):.4f}")
    assert np.abs(hummed).max() > 0.01, "the hum is silent"


def test_the_tune_is_hummed_with_a_closed_mouth(ready_melody=None):
    """The tune render asks for the closed sound, not the open one."""
    from raagacomposer.core.models import MelodyVersion, Note
    from raagacomposer.voice import renderer

    melody = MelodyVersion(notes=[Note(swara="S", midi=60, start=0.0,
                                       duration=0.5)])
    segments = renderer.plan_segments(melody, None, vocal_sections_only=False,
                                      vowel="hum")
    assert segments and all(s.vowel == "hum" for s in segments)
    assert all(s.consonant == "" for s in segments), "a hum has no consonants"


# --------------------------------------------------------------------------
# A syllable is closed, not left open (Krish's fifth walkthrough item)
# --------------------------------------------------------------------------
def _one_note(coda, vowel="aa", sr=44100):
    from raagacomposer.core.models import VocalDirection
    from raagacomposer.voice.profiles import BUILTIN
    from raagacomposer.voice.renderer import SungSegment, render
    seg = SungSegment(start=0.2, end=0.9, midi=60, syllable="x",
                      vowel=vowel, consonant="v", coda=coda)
    return render([seg], BUILTIN[2], VocalDirection(), sr,
                  total_seconds=1.2, seed=3)


def test_a_closing_consonant_reaches_the_audio():
    """The coda was discarded, so "vaan" was sung "vaa"."""
    import numpy as np
    assert not np.allclose(_one_note("n"), _one_note("")), \
        "the closing consonant made no difference to the sound"


def test_different_closing_consonants_sound_different():
    """A coda that renders identically whatever it is has not been read."""
    import numpy as np
    nasal, fricative = _one_note("n"), _one_note("s")
    assert not np.allclose(nasal, fricative)


def test_a_fricative_close_is_brighter_than_a_nasal_one():
    """Not a claim about intelligibility - only that the kinds differ in
    the direction they should.  Both are measured against each other
    rather than against the vowel: replacing tonal energy with filtered
    noise raises a spectral centroid whatever the consonant is."""
    import numpy as np
    sr = 44100

    def centroid(a, lo=0.83, hi=0.90):
        w = a[int(lo * sr):int(hi * sr)] * np.hanning(int((hi - lo) * sr))
        spec = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), 1 / sr)
        return float((spec * freqs).sum() / max(spec.sum(), 1e-12))

    assert centroid(_one_note("s")) > centroid(_one_note("n")) * 1.5


@pytest.mark.parametrize("length", [0.02, 0.05, 0.3])
@pytest.mark.parametrize("coda", ["s", "n", "t", "l"])
def test_a_short_note_does_not_carry_its_coda_into_the_next(length, coda):
    """Arya's finding, and a correction to the test I wrote first.

    _close_with_consonant clamped where the consonant began and left its
    length at the full default, so a twenty-millisecond note wrote a
    seventy-five-millisecond fricative over the note after it.  My own
    boundary test used three-hundred-millisecond notes, long enough that
    the clamp never engaged: it asserted the guarantee in the one case
    that could not break it.

    What is asserted here is what an end-to-end render can honestly show.
    The four resonators carry their state across notes, so changing the
    end of one note does change the beginning of the next - measured at
    about -78 dB, gone within fifty milliseconds.  That is the filter
    ringing out, not a consonant in the wrong place.  A real spill is
    orders of magnitude larger and does not decay.  The buffer-level
    guarantee is asserted directly in the integration boundary tests.
    """
    import numpy as np
    from raagacomposer.core.models import VocalDirection
    from raagacomposer.voice.profiles import BUILTIN
    from raagacomposer.voice.renderer import SungSegment, render
    sr = 44100
    second = 0.2 + length
    pair = [SungSegment(start=0.2, end=second, midi=60, vowel="aa",
                        coda=coda),
            SungSegment(start=second, end=second + 0.3, midi=62, vowel="ee")]
    plain = [SungSegment(start=0.2, end=second, midi=60, vowel="aa"),
             SungSegment(start=second, end=second + 0.3, midi=62, vowel="ee")]
    total = second + 0.4
    with_coda = render(pair, BUILTIN[2], VocalDirection(), sr,
                       total_seconds=total, seed=3)
    without = render(plain, BUILTIN[2], VocalDirection(), sr,
                     total_seconds=total, seed=3)
    difference = np.abs(with_coda - without)

    ring = difference[int(second * sr):int((second + 0.05) * sr)]
    assert ring.max() < 1e-3,         f"{length * 1000:.0f} ms note, {coda!r}: {ring.max():.6f} into the next"
    settled = difference[int((second + 0.05) * sr):]
    assert settled.max() < 1e-5,         f"{coda!r} was still audible {settled.max():.8f} past the next note"


def test_a_coda_stays_inside_its_own_note():
    """A closing consonant must not arrive on top of the next vowel."""
    import numpy as np
    from raagacomposer.core.models import VocalDirection
    from raagacomposer.voice.profiles import BUILTIN
    from raagacomposer.voice.renderer import SungSegment, render
    sr = 44100
    pair = [SungSegment(start=0.2, end=0.5, midi=60, vowel="aa", coda="s"),
            SungSegment(start=0.5, end=0.8, midi=62, vowel="ee")]
    plain = [SungSegment(start=0.2, end=0.5, midi=60, vowel="aa"),
             SungSegment(start=0.5, end=0.8, midi=62, vowel="ee")]
    with_coda = render(pair, BUILTIN[2], VocalDirection(), sr,
                       total_seconds=1.0, seed=3)
    without = render(plain, BUILTIN[2], VocalDirection(), sr,
                     total_seconds=1.0, seed=3)
    after = np.abs(with_coda[int(0.52 * sr):int(0.78 * sr)]
                   - without[int(0.52 * sr):int(0.78 * sr)]).max()
    assert after < 1e-6, f"the coda bled into the next note by {after:.6f}"


# --------------------------------------------------------------------------
# A consonant has a place in the mouth, and F2 moves through it
# --------------------------------------------------------------------------
def _sung(onset="", coda="", vowel="aa", sr=44100):
    from raagacomposer.core.models import VocalDirection
    from raagacomposer.voice.profiles import BUILTIN
    from raagacomposer.voice.renderer import SungSegment, render
    seg = SungSegment(start=0.2, end=0.9, midi=60, vowel=vowel,
                      consonant=onset, coda=coda)
    return render([seg], BUILTIN[2], VocalDirection(), sr,
                  total_seconds=1.1, seed=3)


def _tilt(audio, at, sr=44100, width=0.030):
    """Energy above the vowel's F2 against energy below it.

    Peak-picking cannot see this movement: at a 262 Hz fundamental the
    spectrum is sampled every 262 Hz, so a formant moving a few hundred
    Hz need not move any harmonic - what it moves is the balance between
    them.  A centroid over the whole voice band cannot see it either,
    because F1 carries most of the energy and swamps the change.
    """
    import numpy as np
    w = audio[int((at - width / 2) * sr):int((at + width / 2) * sr)]
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    high = spec[(freqs >= 1400) & (freqs < 2800)].sum()
    low = spec[(freqs >= 600) & (freqs < 1400)].sum()
    return float(high / max(low, 1e-9))


def test_a_lip_consonant_starts_the_vowel_lower():
    """"m" is made at the lips, and F2 comes up from below into the vowel."""
    plain = _tilt(_sung(), 0.212)
    labial = _tilt(_sung("m"), 0.212)
    assert labial < plain * 0.9, f"{labial:.4f} against {plain:.4f}"


def test_a_tongue_consonant_starts_the_vowel_higher():
    plain = _tilt(_sung(), 0.212)
    for onset in ("t", "k", "l"):
        moved = _tilt(_sung(onset), 0.212)
        assert moved > plain * 1.4, f"{onset!r}: {moved:.4f} vs {plain:.4f}"


def test_the_two_places_separate_from_each_other():
    """A transition that exists but cannot be told apart is decoration."""
    assert _tilt(_sung("t"), 0.212) > _tilt(_sung("m"), 0.212) * 2


def test_the_movement_is_over_before_the_note_is():
    """It is a transition, not a different vowel: by the middle of the
    note the sound has arrived and the consonant is no longer shaping it."""
    plain = _tilt(_sung(), 0.55)
    for onset in ("m", "t", "k"):
        assert abs(_tilt(_sung(onset), 0.55) - plain) < plain * 0.05


def test_a_vowel_with_no_consonant_beside_it_does_not_move():
    import numpy as np
    from raagacomposer.voice.renderer import SungSegment, _f2_track
    seg = SungSegment(start=0.0, end=0.7, midi=60, vowel="aa")
    assert _f2_track(seg, 30000, 44100) is None
