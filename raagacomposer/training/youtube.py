"""YouTube as a place to find what to study - never as a source of audio.

The creator's brief is that YouTube is a *curriculum* source: somewhere to
find lessons worth studying, approve them, and turn them into knowledge and
quizzes.  Not somewhere to take music from.  That line is the project's own
(``docs/DECISIONS.md``, "Heard and stated are different kinds of evidence")
and this module sits entirely on the "stated" side of it:

* nothing here downloads a video, an audio track, or a caption file;
* what it produces is a **lead** - a note that material is said to exist at a
  URL - which is the shape ``WebLeadProvider`` already consumes and the
  Training tab already makes the creator approve one at a time;
* the transcript, if the creator wants one studied, is supplied by them.

Two ways to get leads, and the same pattern the rest of the application uses
for anything that could touch a network: the path that needs nothing works
always, and the configured path is an addition.

1. **Paste links.**  Type or paste YouTube URLs into the Training search box
   and each becomes a lead.  No key, no network, works offline.  This is the
   common case: the creator already found the video.
2. **Search by phrase.**  Needs a YouTube Data API key, and is only reached
   when one is configured.  Without it the phrase simply finds nothing here
   and the other providers answer, exactly as they do today.

Titles are looked up through YouTube's public oEmbed endpoint, which returns
a title and a channel and no content at all - and only when
``training_allow_web`` is on, which is the setting that already means "you
may touch the network to find out where things are".  With it off, a lead
carries the URL and the video id and nothing is contacted.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings

log = get_logger("training.youtube")

#: How long to wait on a metadata lookup.  A lead is a convenience; it is
#: never worth making the creator wait.
LOOKUP_TIMEOUT = 4.0

OEMBED = "https://www.youtube.com/oembed"
SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
WATCH = "https://www.youtube.com/watch?v="

#: The forms a YouTube video URL actually takes.  Deliberately strict: an id
#: is eleven characters of a known alphabet, so a stray word is not mistaken
#: for a video.
_PATTERNS = (
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?[^ ]*\bv=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/(?:embed|shorts|v)/([A-Za-z0-9_-]{11})"),
)


def video_ids(text: str) -> List[str]:
    """Every YouTube video id in some text, in order, without repeats."""
    found: List[str] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text or ""):
            if match.group(1) not in found:
                found.append(match.group(1))
    return found


def watch_url(video_id: str) -> str:
    return f"{WATCH}{video_id}"


def describe(video_id: str, allow_network: bool = False) -> Dict[str, str]:
    """Title and channel for a video, or the id when we may not ask.

    oEmbed returns a title, an author and a thumbnail - metadata about where
    something is, which is what a lead is for.  It returns no captions and no
    media.  When ``allow_network`` is false nothing is contacted at all and
    the lead is still perfectly usable; it just says less.
    """
    if not allow_network:
        return {"title": f"YouTube video {video_id}", "author": ""}
    params = urllib.parse.urlencode({"url": watch_url(video_id),
                                     "format": "json"})
    try:
        with urllib.request.urlopen(f"{OEMBED}?{params}",
                                    timeout=LOOKUP_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:                                     # noqa: BLE001
        log.info("no metadata for %s (%s)", video_id, exc.__class__.__name__)
        return {"title": f"YouTube video {video_id}", "author": ""}
    return {"title": str(data.get("title", "") or f"YouTube video {video_id}"),
            "author": str(data.get("author_name", "") or "")}


def leads_from_text(text: str, allow_network: bool = False,
                    limit: int = 10) -> List[Dict[str, Any]]:
    """Leads for every YouTube link in some text.  No key, no search."""
    out: List[Dict[str, Any]] = []
    for video_id in video_ids(text)[:limit]:
        meta = describe(video_id, allow_network)
        out.append({
            "title": meta["title"],
            "url": watch_url(video_id),
            "author": meta["author"],
            "description": "Supplied by you as a link.",
            "video_id": video_id,
        })
    return out


def search_api(phrase: str, limit: int, api_key: str) -> List[Dict[str, Any]]:
    """Search YouTube for candidate lessons.  Only reached with a key."""
    params = urllib.parse.urlencode({
        "part": "snippet", "q": phrase, "type": "video",
        "maxResults": max(1, min(int(limit), 25)), "key": api_key,
    })
    try:
        with urllib.request.urlopen(f"{SEARCH_API}?{params}",
                                    timeout=LOOKUP_TIMEOUT * 2) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:                                     # noqa: BLE001
        log.warning("YouTube search unavailable: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for item in data.get("items", []):
        video_id = str((item.get("id") or {}).get("videoId", ""))
        snippet = item.get("snippet") or {}
        if not video_id:
            continue
        out.append({
            "title": str(snippet.get("title", "")),
            "url": watch_url(video_id),
            "author": str(snippet.get("channelTitle", "")),
            "description": str(snippet.get("description", "")),
            "video_id": video_id,
        })
    return out


def finder(settings: Optional[Settings] = None
           ) -> Callable[[str, int], List[Dict[str, Any]]]:
    """The callable ``WebLeadProvider`` wants: a phrase in, leads out.

    Pasted links are answered without a key or a network call.  A phrase with
    no links in it reaches the Data API only when a key is configured, and
    otherwise finds nothing here - which is not a failure, it is the other
    providers answering instead.
    """
    settings = settings or Settings.load()

    def find(phrase: str, limit: int) -> List[Dict[str, Any]]:
        allow_network = bool(getattr(settings, "training_allow_web", False))
        links = leads_from_text(phrase, allow_network, limit)
        if links:
            return links
        key = Settings.secret("youtube_api_key")
        if not key:
            log.info("no YouTube key configured; pasted links still work")
            return []
        if not allow_network:
            log.info("training_allow_web is off; not searching YouTube")
            return []
        return search_api(phrase, limit, key)

    return find
