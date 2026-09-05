"""The training manifest: which file is which raaga, and what kind.

The two failures this exists to prevent, both seen in a real folder:
one missing letter in a folder name silencing every file in it, and a
talk *about* a raaga being handed to a pitch tracker as if it were a
performance of one.
"""
from __future__ import annotations

import csv

import pytest

from raagacomposer.training import manifest

pytestmark = pytest.mark.unit


def _write(root, rows, name="training_manifest.csv"):
    path = root / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file", "raaga", "kind", "source", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _media(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really audio")
    return path


# --------------------------------------------------------------------------
def test_no_manifest_is_not_an_error(tmp_path):
    assert manifest.find_manifest(tmp_path) is None
    assert manifest.load(tmp_path) == []


def test_a_manifest_above_the_folder_still_governs_it(tmp_path):
    """One index at the top of a collection covers every raaga folder."""
    _media(tmp_path, "Hamsadhwani/mashup.mp3")
    _write(tmp_path, [{"file": "Hamsadhwani/mashup.mp3", "raaga": "Hamsadhwani",
                       "kind": "music", "source": "", "notes": ""}])
    found = manifest.find_manifest(tmp_path / "Hamsadhwani")
    assert found is not None and found.parent == tmp_path
    assert len(manifest.load(tmp_path / "Hamsadhwani")) == 1


def test_the_spelling_of_the_folder_no_longer_decides_anything(tmp_path):
    """A folder called 'hamsadwani' used to silence everything in it."""
    _media(tmp_path, "hamsadwani/mashup.mp3")
    _write(tmp_path, [{"file": "hamsadwani/mashup.mp3", "raaga": "Hamsadhwani",
                       "kind": "music", "source": "", "notes": ""}])
    assert len(manifest.music_for(tmp_path, "Hamsadhwani")) == 1


def test_a_lesson_never_reaches_the_ears(tmp_path):
    """Speech through a pitch tracker is confident nonsense, not silence."""
    _media(tmp_path, "Kalyani/talk.mp3")
    _media(tmp_path, "Kalyani/concert.mp3")
    _write(tmp_path, [
        {"file": "Kalyani/talk.mp3", "raaga": "Kalyani", "kind": "lesson",
         "source": "", "notes": "a talk about the raaga"},
        {"file": "Kalyani/concert.mp3", "raaga": "Kalyani", "kind": "music",
         "source": "", "notes": ""}])

    music = manifest.music_for(tmp_path, "Kalyani")
    assert [e.path.name for e in music] == ["concert.mp3"]
    lessons = manifest.lessons_for(tmp_path, "Kalyani")
    assert [e.path.name for e in lessons] == ["talk.mp3"]


def test_skip_means_skip(tmp_path):
    _media(tmp_path, "Kalyani/dubious.mp3")
    _write(tmp_path, [{"file": "Kalyani/dubious.mp3", "raaga": "Kalyani",
                       "kind": "skip", "source": "", "notes": ""}])
    assert manifest.music_for(tmp_path, "Kalyani") == []


def test_a_video_is_not_ready_until_its_audio_is_extracted(tmp_path):
    """Saying so here beats discovering it half way through an analysis."""
    video = _media(tmp_path, "Hamsadhwani/mashup.mp4")
    _write(tmp_path, [{"file": "Hamsadhwani/mashup.mp4", "raaga": "Hamsadhwani",
                       "kind": "music", "source": "youtube", "notes": ""}])

    assert manifest.music_for(tmp_path, "Hamsadhwani") == []
    assert len(manifest.pending_extraction(tmp_path)) == 1

    extracted = manifest.extracted_path(video)
    extracted.parent.mkdir(parents=True, exist_ok=True)
    extracted.write_bytes(b"wav")
    assert len(manifest.music_for(tmp_path, "Hamsadhwani")) == 1
    assert manifest.pending_extraction(tmp_path) == []


def test_a_stale_row_is_dropped_not_fatal(tmp_path):
    _media(tmp_path, "Kalyani/here.mp3")
    _write(tmp_path, [
        {"file": "Kalyani/gone.mp3", "raaga": "Kalyani", "kind": "music",
         "source": "", "notes": ""},
        {"file": "Kalyani/here.mp3", "raaga": "Kalyani", "kind": "music",
         "source": "", "notes": ""}])
    assert [e.path.name for e in manifest.load(tmp_path)] == ["here.mp3"]


def test_an_unfilled_raaga_matches_nothing_rather_than_everything(tmp_path):
    _media(tmp_path, "misc/song.mp3")
    _write(tmp_path, [{"file": "misc/song.mp3", "raaga": "unknown",
                       "kind": "music", "source": "", "notes": ""}])
    entries = manifest.load(tmp_path)
    assert entries[0].raaga == ""
    assert manifest.music_for(tmp_path, "Hamsadhwani") == []
    assert manifest.music_for(tmp_path, "") == []


def test_an_unknown_kind_is_treated_as_music_and_logged(tmp_path):
    _media(tmp_path, "Kalyani/song.mp3")
    _write(tmp_path, [{"file": "Kalyani/song.mp3", "raaga": "Kalyani",
                       "kind": "banana", "source": "", "notes": ""}])
    assert len(manifest.music_for(tmp_path, "Kalyani")) == 1
