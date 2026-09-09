"""Render identity, versioning and refusals — contract `5ee63d7` §2.

A render is a directory that is written once and never rewritten. Everything
needed to say what produced it lives beside the audio, including the things that
went wrong.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA = "voice-prototype-render/1"

#: What a render may claim about itself. Never more than this.
NOT_HEARD = "not_heard"
LABEL = ("Synthetic reference-conditioned audio. Not an authentic performance by "
         "the target singer, and not an identity or naturalness claim.")


class RenderError(Exception):
    """A refusal. Never a silent correction."""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_render_dir(root: Path, *, parent: str | None = None) -> tuple:
    """A fresh render id and its directory.

    The id carries the clock so renders sort in the order they happened, and a
    counter so two in the same second do not collide. An existing directory is a
    refusal: no render is ever written over another.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for suffix in range(1, 100):
        render_id = f"r{stamp}-{suffix:02d}"
        directory = root / render_id
        if not directory.exists():
            directory.mkdir(parents=True)
            return render_id, directory
    raise RenderError(f"Cannot allocate a render id under {root}.")


class Record:
    """The render record, built as the run proceeds and written at the end."""

    def __init__(self, render_id: str, directory: Path, *, parent: str | None = None):
        self.path = Path(directory) / "render.json"
        self.data = {
            "schema": SCHEMA,
            "render_id": render_id,
            "parent_render_id": parent,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "started",
            "label": LABEL,
            "listening_status": NOT_HEARD,
            "kind": "new_render" if parent is None else "revised_render",
            "inputs": {},
            "stages": {},
            "outputs": {},
            "settings": {},
            "not_held_constant": [],
            "failures": [],
            "warnings": [],
        }

    def input(self, name: str, path=None, **fields) -> None:
        entry = dict(fields)
        if path is not None:
            entry["path"] = str(path)
            entry["sha256"] = sha256(path)
        self.data["inputs"][name] = entry

    def stage(self, name: str, **fields) -> None:
        self.data["stages"][name] = fields

    def output(self, name: str, path, **fields) -> None:
        path = Path(path)
        entry = dict(fields, path=str(path), sha256=sha256(path))
        self.data["outputs"][name] = entry

    def settings(self, **fields) -> None:
        self.data["settings"].update(fields)

    def not_constant(self, field: str, mine, theirs, against: str) -> None:
        """Say plainly what is *not* held constant against a named comparison.

        A comparison that quietly varies a second thing cannot attribute its
        difference to the first.
        """
        self.data["not_held_constant"].append(
            {"field": field, "this_render": mine, "compared_render": theirs,
             "against": against})

    def warn(self, text: str) -> None:
        self.data["warnings"].append(text)

    def fail(self, exc: BaseException) -> None:
        self.data["failures"].append({"type": type(exc).__name__, "detail": str(exc)})
        self.data["status"] = "failed"

    def finish(self, status: str = "rendered_not_accepted", **fields) -> Path:
        if self.data["status"] != "failed":
            self.data["status"] = status
        self.data.update(fields)
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self.write()

    def write(self) -> Path:
        if self.path.exists():
            # The record is written once, at the end, like everything else here.
            raise RenderError(f"{self.path} already exists.")
        staging = self.path.with_name(f"staging-{self.path.name}")
        with staging.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, self.path)
        return self.path


def measure_audio(path) -> dict:
    """What the file is, measured. Never asserted through a tolerance.

    A check that admits full scale is not a clipping check, so the rail count is
    reported as a number and left for a person to judge.
    """
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    at_rail = int(np.sum(np.abs(audio) >= 0.999999))
    measured = {
        "seconds": round(len(audio) / rate, 6),
        "sample_rate": rate,
        "peak": round(float(np.max(np.abs(audio))), 6),
        "rms_dbfs": round(float(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))),
                                                  1e-12))), 3),
        "samples_at_rail": at_rail,
        "finite": bool(np.isfinite(audio).all()),
    }
    if at_rail:
        measured["rail_note"] = (
            f"{at_rail} sample(s) reach full scale. This file is not "
            f"clipping-free, and lowering its gain afterwards cannot undo it - "
            f"only a re-render with more headroom can.")
    return measured


def listening_copy(source, destination, *, target_dbfs: float = -18.0) -> dict:
    """A level-matched copy for listening. Gain only.

    No EQ, no limiting, no resampling, no re-render. If the target level will
    not fit, the pull-back is stated rather than silently limited.
    """
    import numpy as np
    import soundfile as sf

    destination = Path(destination)
    if destination.exists():
        raise RenderError(f"{destination} exists; copies are not overwritten.")
    audio, rate = sf.read(str(source), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms <= 0:
        raise RenderError(f"{source} is silent; there is nothing to level.")
    gain = 10 ** (target_dbfs / 20) / rms
    scaled = audio * gain
    extra = 1.0
    peak = float(np.max(np.abs(scaled)))
    if peak >= 0.999:
        extra = 0.999 / peak
        scaled *= extra
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), scaled, rate, subtype="PCM_24")
    return {
        "from": str(source), "to": str(destination), "to_sha256": sha256(destination),
        "target_rms_dbfs": target_dbfs,
        "gain_db": round(float(20 * np.log10(gain * extra)), 3),
        "extra_headroom_applied": extra != 1.0,
        "source_samples_at_rail": int(np.sum(np.abs(audio) >= 0.999999)),
        "processing": "gain only; no EQ, limiting, resampling or re-render",
        "note": ("If the source had samples at the rail, this copy is quieter, "
                 "not repaired."),
    }
