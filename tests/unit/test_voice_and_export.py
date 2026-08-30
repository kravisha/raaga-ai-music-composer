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
    assert split_syllable(syllable) == (consonant, vowel)


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
