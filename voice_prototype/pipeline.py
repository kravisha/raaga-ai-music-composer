"""Tamil lyrics + a tune + a target reference → one sung audio file.

Contract `46e5ca2` (v2). The whole prototype, in one place, so a reader can see
what happens in what order and what each step refuses.

Reference, tune, lyrics, output and feedback are linked by identity and hash in
the render record — §1.6 — rather than by sharing a folder.

**The target reference is used at inference only.** No training, no corpus
admission, and no claim about the singer. Everything it produces is labelled
synthetic reference-conditioned audio.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import align, carrier, render as render_mod, tune as tune_mod

#: The Tamil rules live in one place. Imported by path rather than copied, so
#: there is one source of truth for them, and hashed into every render record.
TAMIL_FRONTEND = Path(r"C:\Users\Krish\ClaudeStuff\Claude-Tamil-Frontend-20260909"
                      r"\tamil_frontend.py")


class PrototypeError(Exception):
    """A refusal. Never a silent correction."""


def load_tamil_frontend(path: Path = TAMIL_FRONTEND):
    path = Path(path)
    if not path.is_file():
        raise PrototypeError(
            f"The Tamil frontend is not at {path}. It is imported rather than "
            f"copied so the rules have one home; fix the path rather than "
            f"duplicating it.")
    spec = importlib.util.spec_from_file_location("tamil_frontend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("tamil_frontend", module)
    spec.loader.exec_module(module)
    return module


@dataclass
class Reference:
    """The target voice sample, with the record that authorized it.

    Its limits travel with every render made from it. §1.1, §1.5.
    """
    path: Path
    sha256: str
    use_scope: str
    limits: list = field(default_factory=list)
    parent_source: str | None = None
    interval_s: list | None = None
    is_isolated_vocal: bool = False

    def verify(self) -> dict:
        path = Path(self.path)
        if not path.is_file():
            raise PrototypeError(f"The reference {path} is not present.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.sha256:
            raise PrototypeError(
                f"{path} is {actual[:12]}, not the recorded {self.sha256[:12]}. "
                f"A different file under the same name is a different voice.")
        return {
            "path": str(path), "sha256": actual, "use_scope": self.use_scope,
            "limits": list(self.limits),
            "parent_source": self.parent_source,
            "interval_s": self.interval_s,
            "is_isolated_vocal": bool(self.is_isolated_vocal),
            "isolation_note": (
                "Recorded as an isolated vocal stem." if self.is_isolated_vocal else
                "NOT an isolated vocal stem. This is a mix, and it is not "
                "relabelled because a later stage would prefer otherwise."),
            "used_for": "inference conditioning only; not training, not admission",
        }


@dataclass
class Bank:
    """The guide voice bank and the engine that plays it."""
    zip_path: Path
    dll_path: Path

    def aliases(self) -> set:
        bank = zipfile.ZipFile(str(self.zip_path))
        found = set()
        for line in bank.read("oto.ini").decode("utf-8-sig").splitlines():
            if "=" not in line:
                continue
            filename, values = line.split("=", 1)
            found.add(values.split(",")[0] or Path(filename).stem)
        return found


def sing(*, lyrics: str, slots: list, melody: tune_mod.Tune, reference: Reference,
         bank: Bank, out_root: Path, convert=None, parent_render: str | None = None,
         decided_by: str = "", notes: str = "") -> dict:
    """One iteration, start to finish.

    `slots` carries the Tamil per articulation and which slots begin a word;
    `melody` carries pitch, duration and voicing. Both are inputs a person wrote,
    which is the honest position - none of this infers a melody or a phrasing.
    """
    if not decided_by:
        raise PrototypeError("A render names who asked for it.")
    frontend = load_tamil_frontend()

    render_id, directory = render_mod.new_render_dir(out_root, parent=parent_render)
    record = render_mod.Record(render_id, directory, parent=parent_render)
    record.data["contract"] = "docs/spec/VOICE_PROTOTYPE_CONTRACT.md at 46e5ca2 (v2)"
    record.data["requested_by"] = decided_by
    record.data["notes"] = notes

    try:
        # -- inputs, all five linked by identity ---------------------------
        lyric_path = directory / "lyrics.txt"
        lyric_path.write_text(lyrics, encoding="utf-8")
        record.input("lyrics", path=lyric_path, text=lyrics, script="Tamil",
                     convention="Tamil script as written; not a romanisation")
        tune_path = melody.save(directory / "tune.json")
        record.input("tune", path=tune_path, source=melody.source,
                     is_estimate=melody.is_estimate,
                     seconds=round(melody.duration_s, 6),
                     sung_events=len(melody.sung_events),
                     rests=len([e for e in melody.events if e.get("midi") is None]))
        record.input("reference", **reference.verify())
        record.input("tamil_frontend", path=TAMIL_FRONTEND)
        record.input("voice_bank", path=bank.zip_path,
                     voice="Hanakoloid CVVC by HanadaP - the guide, not the target")

        # -- lyrics -> alias requests; refuses by name ----------------------
        requests = frontend.slot_requests(slots, bank.aliases())
        syllables = [frontend.parse(s["draft_articulation"])[0] for s in slots]
        for syllable, slot in zip(syllables, slots):
            syllable.text = slot["draft_articulation"]
        record.stage("lyrics", syllables=len(syllables),
                     requests=[[head, tail] for head, tail in requests],
                     refusal_policy="unsupported sounds are refused by name")

        # -- alignment: no overflow, rests stay rests -----------------------
        placed = align.place(syllables, melody)
        align.save(placed, directory / "alignment.json")
        record.stage("alignment", placed=len(placed),
                     phrases=len({p.phrase for p in placed}),
                     ends_at_s=round(max(p.end_s for p in placed), 6),
                     tune_ends_at_s=round(melody.duration_s, 6))

        # -- carrier --------------------------------------------------------
        guide_path = directory / "guide.wav"
        guide = carrier.render(placed, melody, requests, bank_zip=bank.zip_path,
                               dll_path=bank.dll_path, destination=guide_path)
        record.stage("carrier", **{k: v for k, v in guide.items() if k != "events"})
        (directory / "carrier-events.json").write_text(
            json.dumps(guide["events"], indent=2), encoding="utf-8")
        record.output("guide", guide_path, **render_mod.measure_audio(guide_path))

        # -- conversion to the target reference -----------------------------
        if convert is None:
            record.warn("No conversion backend supplied: the guide is the only "
                        "audio produced, and a carrier alone does not complete "
                        "the milestone.")
        else:
            converted = convert(guide_path, reference, directory, record)
            record.output("converted", converted,
                          **render_mod.measure_audio(converted))
            copy = render_mod.listening_copy(
                converted, directory / "listening" / "B-converted-minus18.wav")
            record.output("listening_converted", copy["to"], **copy)
        guide_copy = render_mod.listening_copy(
            guide_path, directory / "listening" / "A-guide-minus18.wav")
        record.output("listening_guide", guide_copy["to"], **guide_copy)

    except Exception as exc:                       # noqa: BLE001 - recorded
        record.fail(exc)
        record.finish(status="failed")
        raise

    record.finish()
    return {"render_id": render_id, "directory": str(directory),
            "record": str(record.path),
            "guide": str(guide_path),
            "converted": record.data["outputs"].get("converted", {}).get("path"),
            "complete": "converted" in record.data["outputs"]}
