"""Extract the audio from the videos a training manifest calls music.

The corpus reader hands audio to the ears; a downloaded collection is
video.  This decodes each ``music`` entry once into
``<folder>/.extracted/<name>.wav`` at the analysis sample rate, so a
folder of mp4s is decoded on the way in rather than on every learning
pass.

``lesson`` entries are deliberately skipped.  Their audio is speech, and
speech handed to a pitch tracker yields a confident contour of nothing.
They belong to the watch track, as transcripts.

Usage::

    python tools/extract_audio.py <training folder> [--dry-run] [--force]

Requires ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raagacomposer.agent import analysis                      # noqa: E402
from raagacomposer.training import manifest                   # noqa: E402


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit(
            "ffmpeg is not on PATH.  Install it (winget install "
            "Gyan.FFmpeg) and open a new shell, then run this again.")
    return found


def extract(exe: str, video: Path, target: Path, sample_rate: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Mono at the analysis rate: the pitch tracker wants one channel, and
    # writing it here means every later pass reads it ready to use.
    command = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-f", "wav", str(target)]
    subprocess.run(command, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="the training folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be extracted, do nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-extract even when the wav already exists")
    args = parser.parse_args(argv)

    entries = manifest.load(args.folder)
    if not entries:
        print(f"no manifest governing {args.folder}")
        return 1

    music = [e for e in entries if e.is_music]
    lessons = [e for e in entries if e.is_lesson]
    videos = [e for e in music
              if e.path.suffix.lower() in manifest.VIDEO_SUFFIXES]

    print(f"{len(entries)} entr(ies): {len(music)} music, {len(lessons)} lesson")
    print(f"{len(videos)} music video(s) to extract")
    if lessons:
        print(f"skipping {len(lessons)} lesson(s) - speech is for the watch "
              f"track, not the ears")

    if args.dry_run:
        for entry in videos:
            target = manifest.extracted_path(entry.path)
            state = "exists" if target.exists() else "would extract"
            print(f"  [{state}] {entry.path.name}")
        return 0

    exe = ffmpeg()
    done = skipped = 0
    for entry in videos:
        target = manifest.extracted_path(entry.path)
        if target.exists() and not args.force:
            skipped += 1
            continue
        print(f"  extracting {entry.path.name}")
        try:
            extract(exe, entry.path, target, analysis.DEFAULT_SR)
        except subprocess.CalledProcessError as exc:
            print(f"    FAILED ({exc.returncode})")
            continue
        done += 1

    print(f"extracted {done}, already present {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
