"""Small, explicit-path production journal beside an existing song project.

Artifact references point to the composer's existing versions; audio and the
knowledge database are not duplicated or opened here. Invalidation preserves
review history while making prior acceptance inapplicable to the new revision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

PRODUCTION_STAGES = ("brief", "tune", "lyrics", "voice", "beat", "arrangement", "mix")
_REFS = {"tune": "tune_ref", "lyrics": "lyric_ref", "voice": "vocal_takes",
         "beat": "beat", "arrangement": "arrangement", "mix": "mix"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Reject objects/NaN instead of silently stringifying evidence."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _stage(stage: str) -> None:
    if stage not in PRODUCTION_STAGES:
        raise ValueError(f"Unknown production stage: {stage}")


@dataclass(frozen=True)
class StageRecord:
    stage: str
    role: str
    artifact_ref: str
    rationale: str
    verdict: str
    round: int
    at: str = field(default_factory=utc_now)
    revision: int = 0
    provider: str = ""
    evidence: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    model: str = ""

    def __post_init__(self) -> None:
        _stage(self.stage)
        if self.verdict not in ("accept", "revise", "blocked"):
            raise ValueError("Invalid stage verdict")
        if type(self.round) is not int or self.round < 1:
            raise ValueError("Review rounds start at 1")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("Invalid project revision")
        for key in ("role", "artifact_ref", "rationale", "at", "provider", "model"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be text")
        if not self.role.strip() or not self.artifact_ref.strip() or not self.rationale.strip():
            raise ValueError("A review needs a role, artifact and rationale")
        for key in ("evidence", "revisions", "strengths", "lessons", "uncertainty"):
            value = getattr(self, key)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValueError(f"{key} must be a list of text values")
        if datetime.fromisoformat(self.at).tzinfo is None:
            raise ValueError("Review timestamps must identify their time zone")


@dataclass
class ProjectState:
    project_id: str = ""
    brief: dict[str, Any] = field(default_factory=dict)
    tune_ref: str = ""
    lyric_ref: str = ""
    vocal_takes: list[str] = field(default_factory=list)
    beat: str = ""
    arrangement: str = ""
    mix: str = ""
    records: list[StageRecord] = field(default_factory=list)
    revision: int = 0
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Validate at the persistence boundary, including mutable fields.
        self.from_dict(data)
        return {"schema_version": 1, **data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectState:
        if not isinstance(data, dict):
            raise ValueError("Production state must be an object")
        value = dict(data)
        if value.pop("schema_version", 1) != 1:
            raise ValueError("Unsupported production state schema")
        allowed = {f.name for f in fields(cls)}
        if set(value) - allowed:
            raise ValueError("Unknown production state fields")
        if type(value.get("revision", 0)) is not int or value.get("revision", 0) < 0:
            raise ValueError("Invalid project revision")
        for key in ("project_id", "tune_ref", "lyric_ref", "beat", "arrangement", "mix"):
            if not isinstance(value.get(key, ""), str):
                raise ValueError(f"{key} must be an artifact reference or text")
        if not isinstance(value.get("brief", {}), dict):
            raise ValueError("Brief must be an object")
        for key in ("vocal_takes", "feedback"):
            items = value.get(key, [])
            if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
                raise ValueError(f"{key} must be a list of text values")
        raw_records = value.get("records", [])
        if not isinstance(raw_records, list):
            raise ValueError("Records must be a list")
        canonical_json(value)
        value = json.loads(canonical_json(value))  # Own all mutable input values.
        value["records"] = [StageRecord(**record) for record in value.get("records", [])]
        if any(record.revision > value.get("revision", 0) for record in value["records"]):
            raise ValueError("Review belongs to a future project revision")
        return cls(**value)

    def fingerprint(self) -> str:
        data = self.to_dict()
        data.pop("records")  # Appending another review is not a musical edit.
        return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()

    def add_record(self, record: StageRecord) -> None:
        historical_failure = record.revision < self.revision and record.verdict == "blocked"
        if record.revision != self.revision and not historical_failure:
            raise ValueError("Cannot accept a review of a different revision")
        self.records.append(StageRecord(**json.loads(canonical_json(asdict(record)))))

    def is_accepted(self, stage: str, artifact_ref: str) -> bool:
        _stage(stage)
        matching = [r for r in self.records if r.stage == stage and r.role == "critic"
                    and r.revision == self.revision]
        return bool(matching and matching[-1].artifact_ref == artifact_ref
                    and matching[-1].verdict == "accept" and matching[-1].provider == "codex")

    def invalidate_from(self, stage: str, reason: str) -> None:
        _stage(stage)
        if not reason.strip():
            raise ValueError("Explain why the production changed")
        self.revision += 1
        for affected in PRODUCTION_STAGES[PRODUCTION_STAGES.index(stage):]:
            if affected in _REFS:
                setattr(self, _REFS[affected], [] if affected == "voice" else "")
        self.feedback.append(reason)

    def save(self, path: str | Path) -> Path:
        """Atomic journal replacement, preserving history and rejecting concurrent writers."""
        target = Path(path)
        payload = self.to_dict()
        if not self.project_id.strip():
            raise ValueError("A saved production journal needs a project identity")
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.with_name(target.name + ".lock")
        # Never remove someone else's lock, even if its writer disappeared.
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        temporary = None
        try:
            if target.exists():
                current_bytes = target.read_bytes()
                if (getattr(self, "_saved_path", None) != target.resolve()
                    or getattr(self, "_saved_digest", None) != hashlib.sha256(current_bytes).hexdigest()):
                    # Parse first so damaged preexisting journals are explicitly reported.
                    self.from_dict(json.loads(current_bytes.decode("utf-8-sig")))
                    raise ValueError("Refusing a stale journal; reload its preserved history before saving")
                previous = self.load(target)
                old_records = previous.to_dict()["records"]
                if previous.project_id != self.project_id or previous.revision > self.revision:
                    raise ValueError("Refusing to overwrite another project or newer revision")
                if payload["records"][:len(old_records)] != old_records:
                    raise ValueError("Refusing to discard preserved production history")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                    prefix=target.name + ".", suffix=".tmp", dir=target.parent,
                    delete=False) as output:
                temporary = Path(output.name)
                json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            temporary = None
            self._saved_path = target.resolve()
            self._saved_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink()
        return target

    @classmethod
    def load(cls, path: str | Path) -> ProjectState:
        target = Path(path)
        data = target.read_bytes()
        state = cls.from_dict(json.loads(data.decode("utf-8-sig")))
        state._saved_path = target.resolve()
        state._saved_digest = hashlib.sha256(data).hexdigest()
        return state
