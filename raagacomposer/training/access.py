"""Phase A - what of this source can we honestly reach?

Specification sections 4 and 7.  This is the gate that keeps the rest of the
pipeline truthful.  It decides, before anything is fetched, which
representation of a source we are actually entitled and able to use, and it
records that decision so the report can say so.

The rule it enforces is section 20 rule 10: the system does not bypass
technical or legal access controls.  There is no code here that logs in,
follows a paywall, strips DRM, or downloads from a platform that has not
handed the file over.  A source we cannot reach is marked as such and the
creator is offered the two honest ways forward - supply the file, or supply a
transcript.  Nothing here ever upgrades a source's accessibility on its own
say-so; only a creator handing over a file can do that.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from ..core.logging_setup import get_logger
from .models import Accessibility, LearningSource
from .store import TrainingStore

log = get_logger("training.access")

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
TEXT_SUFFIXES = {".txt", ".md", ".srt", ".vtt", ".json"}
#: Containers we can name but not decode: there is no demuxer in the
#: application, so the audio track has to be extracted by the creator first.
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

#: Schemes the application serves itself.
INTERNAL_SCHEMES = {"raaga-exercise"}


class AccessDecision:
    """What we may use, and what we should tell the creator."""

    def __init__(self, status: str, representation: str = "",
                 reason: str = "", offers: Optional[List[str]] = None) -> None:
        self.status = status
        #: audio | transcript | exercise | none
        self.representation = representation
        self.reason = reason
        self.offers = offers or []

    @property
    def analysable(self) -> bool:
        return self.representation in ("audio", "transcript", "exercise")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AccessDecision({self.status}, {self.representation!r}, "
                f"{self.reason!r})")


class SourceAccessService:
    """Phase A: verify, classify, and decide what may be used."""

    def __init__(self, store: TrainingStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    def check(self, source: LearningSource) -> AccessDecision:
        # 1. A file the creator handed us outranks everything: it is theirs,
        #    it is local, and no access control stands between us and it.
        if source.local_path:
            decision = self._check_local(Path(source.local_path))
            if decision is not None:
                return decision

        if not source.url:
            return AccessDecision(
                Accessibility.NOT_ACCESSIBLE, "",
                "the source has no location we can check",
                ["Upload the source file manually", "Provide transcript"])

        parsed = urlparse(source.url)
        scheme = (parsed.scheme or "").lower()

        # 2. Material the application renders for itself.
        if scheme in INTERNAL_SCHEMES:
            return AccessDecision(
                Accessibility.ACCESSIBLE, "exercise",
                "rendered by the application from its own raaga library")

        # 3. A local file named by URL.
        if scheme == "file":
            path = Path(parsed.path)
            decision = self._check_local(path)
            if decision is not None:
                return decision
            return AccessDecision(
                Accessibility.NOT_ACCESSIBLE, "",
                f"{path} is not readable",
                ["Upload the source file manually"])

        # 4. Anything on the network.  We do not fetch it.  If the creator has
        #    supplied a transcript alongside it we may read that; otherwise it
        #    is a lead, and the report will say the content was not analysed.
        if scheme in ("http", "https"):
            transcript = self._supplied_transcript(source)
            if transcript is not None:
                return AccessDecision(
                    Accessibility.TRANSCRIPT, "transcript",
                    "using the transcript supplied for this source")
            return AccessDecision(
                Accessibility.METADATA_ONLY, "",
                "only the public description is available; the content itself "
                "has not been fetched and has not been analysed",
                ["Upload the source file manually", "Provide transcript"])

        return AccessDecision(
            Accessibility.UNSUPPORTED, "",
            f"nothing here can open a '{scheme}' source",
            ["Upload the source file manually", "Provide transcript"])

    # ------------------------------------------------------------------
    @staticmethod
    def _check_local(path: Path) -> Optional[AccessDecision]:
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            # Being honest about a real limitation rather than failing later
            # with something that looks like a bug.
            return AccessDecision(
                Accessibility.USER_FILE_REQUIRED, "",
                f"{suffix} is a video container and this application has no "
                f"demuxer; extract the audio track and supply that",
                ["Upload the source file manually"])
        if not path.exists() or not path.is_file():
            return None
        if suffix in AUDIO_SUFFIXES:
            return AccessDecision(Accessibility.ACCESSIBLE, "audio",
                                  f"reading {path.name}")
        if suffix in TEXT_SUFFIXES:
            return AccessDecision(Accessibility.TRANSCRIPT, "transcript",
                                  f"reading {path.name}")
        return AccessDecision(
            Accessibility.UNSUPPORTED, "",
            f"'{suffix or path.name}' is not a format this can read",
            ["Upload the source file manually"])

    @staticmethod
    def _supplied_transcript(source: LearningSource) -> Optional[str]:
        text = str(source.metadata.get("transcript", "") or "").strip()
        return text or None

    # ------------------------------------------------------------------
    def already_learned(self, source: LearningSource):
        """Section 10 - the completed run for this lesson, if there is one."""
        return self.store.completed_run_for(source)

    def record_provenance(self, source: LearningSource, run_id: str,
                          decision: AccessDecision) -> None:
        """Section 16 - what we decided, and why, before anything was read."""
        self.store.audit(
            "access.checked",
            f"{decision.status}: {decision.reason}",
            run_id=run_id, source_id=source.source_id)
        self.store.update_candidate(
            source.source_id, accessibility_status=decision.status)
