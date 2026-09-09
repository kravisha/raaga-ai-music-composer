"""Durable feedback — contract `5ee63d7` §3.

Bound to the bytes that were actually heard. Three rules do most of the work:

* **A verdict on one render is a verdict on that render.** It is recorded
  against the render id *and* the audio hash, so if the file changes the
  feedback is reported as stale rather than quietly carried forward.
* **Viewing is not approving.** Absence of feedback is absence of feedback, and
  a render with none is *not heard*, not *acceptable*.
* **A comment is not a weight.** This records what a person said. It changes no
  model, and nothing here may be presented as having trained anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA = "voice-prototype-feedback/1"

PREFERRED = "preferred"
NOT_PREFERRED = "not_preferred"
NO_PREFERENCE = "no_preference"
NEEDS_WORK = "needs_work"
#: No preference is an answer with its own meaning, not a missing one.
DISPOSITIONS = (PREFERRED, NOT_PREFERRED, NO_PREFERENCE, NEEDS_WORK)

#: Asked apart, because one forced choice conflates them and they can move in
#: opposite directions.
ASPECTS = ("words", "ornaments", "rests_and_phrasing", "voice_resemblance",
           "artifacts_or_ringing")


class FeedbackError(Exception):
    """A refusal. Never a silent correction."""


class Ledger:
    """Every comment anyone has made, and what it was made about."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list = []
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("schema") != SCHEMA:
                raise FeedbackError(
                    f"Feedback schema {loaded.get('schema')!r} is not {SCHEMA!r}.")
            self.entries = list(loaded.get("entries") or [])
            for entry in self.entries:
                if entry.get("disposition") not in DISPOSITIONS:
                    raise FeedbackError(
                        f"Stored disposition {entry.get('disposition')!r} is not "
                        f"one of {DISPOSITIONS}.")

    def add(self, *, render_id: str, audio_path, author: str, text: str,
            disposition: str, aspects: dict | None = None,
            segment: str | None = None) -> dict:
        if not author:
            raise FeedbackError("Feedback names who gave it.")
        if not text or not text.strip():
            raise FeedbackError(
                "Feedback carries what was said. An empty comment is not a verdict.")
        if disposition not in DISPOSITIONS:
            raise FeedbackError(
                f"{disposition!r} is not one of {DISPOSITIONS}. No preference is a "
                f"real answer; there is no value here for 'not yet heard', because "
                f"that is the absence of an entry.")
        unknown = sorted(set(aspects or {}) - set(ASPECTS))
        if unknown:
            raise FeedbackError(f"Unknown aspects {unknown}; expected {list(ASPECTS)}.")
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FeedbackError(
                f"{audio_path} is not present, so this feedback cannot be bound to "
                f"the bytes it is about.")
        entry = {
            "entry_id": f"f{len(self.entries) + 1:04d}",
            "render_id": render_id,
            "audio": str(audio_path),
            "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "author": author,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "text": text.strip(),
            "segment": segment,
            "disposition": disposition,
            "aspects": dict(aspects or {}),
            "note": ("A record of what a person said about these exact bytes. It "
                     "updates no model weights."),
        }
        self.entries.append(entry)
        return entry

    def for_render(self, render_id: str) -> list:
        return [e for e in self.entries if e["render_id"] == render_id]

    def status(self, render_id: str, audio_path=None) -> dict:
        """What is actually known about this render, and nothing more."""
        entries = self.for_render(render_id)
        if not entries:
            return {"render_id": render_id, "heard": False,
                    "disposition": None,
                    "note": ("No feedback recorded. Not heard - which is not the "
                             "same as acceptable.")}
        stale = []
        if audio_path is not None and Path(audio_path).is_file():
            current = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
            stale = [e["entry_id"] for e in entries if e["audio_sha256"] != current]
        latest = entries[-1]
        result = {"render_id": render_id, "heard": True,
                  "disposition": latest["disposition"],
                  "entries": len(entries), "latest": latest["entry_id"]}
        if stale:
            result["stale_entries"] = stale
            result["note"] = ("Some feedback was given on different bytes than the "
                              "file now holds; it describes a render that no longer "
                              "exists at this path.")
        return result

    def save(self) -> Path:
        payload = {"schema": SCHEMA, "entries": self.entries,
                   "note": ("Feedback only. Nothing here trains a model or changes "
                            "any weights.")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.path.with_name(f"staging-{self.path.name}")
        with staging.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, self.path)
        return self.path
