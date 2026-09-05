"""A training folder's manifest: which file is which raaga, and what kind.

Until now the corpus provider decided a file's raaga by looking for the
raaga's name inside the filename or its parent folder.  That is fragile in
two ways a real folder exposes immediately.

One missing letter silences everything: a folder spelled ``hamsadwani``
never matches ``Hamsadhwani``, and nothing reports it - the agent simply
finds no material and learns nothing.  And a folder name can only say one
thing, when a downloaded collection says several: *this* file is a
performance in Kalyani, *that* one is a talk about Kalyani, and this third
one is a mashup that moves through two raagas.

A talk is the important case.  Speech handed to a pitch tracker does not
fail loudly; it produces a confident contour of nothing, and those phrases
would reach the composer.  ``kind`` keeps the two apart:

``music``   goes to the ears - audio analysis, phrases the composer may use
``lesson``  goes to the watch track - what a person *said*, a transcript,
            capped at "can explain" and never fed to the pitch tracker
``skip``    ignored

The manifest is a CSV so it can be corrected in a text editor or a
spreadsheet, without renaming a single media file.  It is optional: with
no manifest present the old name-matching still applies.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core.logging_setup import get_logger

log = get_logger("training.manifest")

MANIFEST_NAMES = ("training_manifest.csv", "manifest.csv")

#: Where extracted audio is cached beside a video, so a 1.3GB folder of
#: mp4s is decoded once rather than on every learning pass.
EXTRACTED_DIR = ".extracted"

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".m4v", ".mov"}

KINDS = ("music", "lesson", "skip")


@dataclass(frozen=True)
class Entry:
    """One row: a media file, what it is, and what it is of."""

    path: Path
    raaga: str
    kind: str = "music"
    source: str = ""
    notes: str = ""

    @property
    def is_music(self) -> bool:
        return self.kind == "music"

    @property
    def is_lesson(self) -> bool:
        return self.kind == "lesson"

    def audio_path(self) -> Optional[Path]:
        """The file to hand the ears, or ``None`` if there is not one yet.

        An audio file is itself.  A video's audio has to be extracted
        first (``tools/extract_audio.py``); until that has happened this
        returns ``None`` rather than pretending the material is ready.
        """
        suffix = self.path.suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            return self.path if self.path.exists() else None
        if suffix in VIDEO_SUFFIXES:
            extracted = extracted_path(self.path)
            return extracted if extracted.exists() else None
        return None


def extracted_path(video: Path) -> Path:
    """Where ``video``'s audio lives once extracted."""
    return video.parent / EXTRACTED_DIR / (video.stem + ".wav")


def find_manifest(root: Path) -> Optional[Path]:
    """The manifest governing ``root``, looked for in it and above it.

    Looking upward means a per-raaga subfolder is covered by one manifest
    at the top of the training folder, which is how a collection actually
    grows: many folders, one index.
    """
    root = Path(root)
    for folder in (root, *root.parents):
        for name in MANIFEST_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
        # Do not walk the whole disk looking for one.
        if folder.parent == folder:
            break
    return None


def load(root: Path) -> List[Entry]:
    """Every entry of the manifest governing ``root``, or ``[]``.

    Paths are relative to the manifest's own folder, so the file can be
    moved with its collection.  A row naming a file that is not there is
    logged and dropped - a stale row should not stop the rest working.
    """
    manifest = find_manifest(Path(root))
    if manifest is None:
        return []

    base = manifest.parent
    entries: List[Entry] = []
    try:
        with manifest.open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), start=2):
                entry = _row_to_entry(row, base, manifest, number)
                if entry is not None:
                    entries.append(entry)
    except OSError as exc:
        log.warning("could not read %s: %s", manifest, exc)
        return []

    log.info("manifest %s: %d entr(ies), %d music, %d lesson", manifest,
             len(entries), sum(1 for e in entries if e.is_music),
             sum(1 for e in entries if e.is_lesson))
    return entries


def _row_to_entry(row: Dict[str, str], base: Path, manifest: Path,
                  number: int) -> Optional[Entry]:
    name = (row.get("file") or "").strip()
    if not name:
        return None
    path = (base / name).resolve()
    if not path.is_file():
        log.warning("%s line %d: no such file: %s", manifest.name, number, name)
        return None

    kind = (row.get("kind") or "music").strip().lower()
    if kind not in KINDS:
        log.warning("%s line %d: unknown kind %r, treating as music",
                    manifest.name, number, kind)
        kind = "music"

    raaga = (row.get("raaga") or "").strip()
    if raaga.lower() in ("", "unknown", "?"):
        raaga = ""

    return Entry(path=path, raaga=raaga, kind=kind,
                 source=(row.get("source") or "").strip(),
                 notes=(row.get("notes") or "").strip())


def music_for(root: Path, raaga: str) -> List[Entry]:
    """Entries the ears may listen to for ``raaga``.

    Three things have to be true: the manifest calls it music, it is of
    this raaga, and its audio is actually on disk.  An entry that names a
    video nobody has extracted yet is not ready, and saying so here keeps
    the caller from discovering it half way through an analysis.
    """
    wanted = raaga.strip().lower()
    if not wanted:
        return []
    return [e for e in load(root)
            if e.is_music and e.raaga.lower() == wanted and e.audio_path()]


def lessons_for(root: Path, raaga: str) -> List[Entry]:
    """Entries for the watch track - what was said, never analysed."""
    wanted = raaga.strip().lower()
    if not wanted:
        return []
    return [e for e in load(root)
            if e.is_lesson and e.raaga.lower() == wanted]


def pending_extraction(root: Path) -> List[Entry]:
    """Music entries that are video with no audio extracted yet."""
    return [e for e in load(root)
            if e.is_music and e.path.suffix.lower() in VIDEO_SUFFIXES
            and not extracted_path(e.path).exists()]
