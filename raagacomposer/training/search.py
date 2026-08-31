"""Finding material worth learning from - specification section 5.

One service, several providers, one normalised result shape.  The Training tab
never learns what a provider is: it asks for candidates and gets
:class:`LearningSource` objects, which is what lets a new kind of source be
added later without the tab, the queue or the report changing.

Three providers ship.  Two of them can actually be *learned from* rather than
merely listed:

``exercises``   material the agent renders from its own structural library and
                then listens to with its own ears.  Always available, needs no
                network and raises no rights question at all.
``library``     audio the creator put in their own learning folder.  Theirs,
                and therefore analysable.
``web``         off by default.  When enabled it records *leads* - a note that
                material is said to exist somewhere - marked metadata-only, and
                fetches nothing.  Section 4 forbids circumventing any technical
                or legal access control, and a lead is the honest shape of a
                source we may not take.

Ranking is by usefulness *for learning*, which is not the same as textual
relevance: a source whose content we can actually analyse outranks one we can
only name, because the second cannot teach anybody anything.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from ..raaga.library import Raaga, RaagaLibrary
from .models import Accessibility, LearningSource, SearchQuery
from .store import TrainingStore, identity_of

log = get_logger("training.search")

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}

STOP_WORDS = {
    "a", "an", "and", "for", "how", "in", "is", "lesson", "lessons", "of",
    "on", "the", "to", "tutorial", "tutorials", "video", "videos", "with",
    "learn", "learning", "class", "classes",
}

#: Topics the exercise provider can actually build a lesson for, with the
#: words that suggest each one.  The lesson kinds are the agent's own: these
#: are things it can render, hear and be marked on.
EXERCISE_TOPICS: Sequence[Dict[str, Any]] = (
    {"key": "identity", "title": "{raaga}: identity and scale",
     "goal": "arohanam, avarohanam and the swaras that belong to the raaga",
     "difficulty": "beginner",
     "words": ("arohanam", "avarohanam", "scale", "swara", "swaras",
               "identity", "beginner", "basic", "introduction", "notes")},
    {"key": "prayoga", "title": "{raaga}: characteristic phrases",
     "goal": "the prayogas that make the raaga recognisable",
     "difficulty": "intermediate",
     "words": ("prayoga", "prayogas", "phrase", "phrases", "sanchara",
               "characteristic", "identity", "idiom")},
    {"key": "gamaka", "title": "{raaga}: gamaka and ornamentation",
     "goal": "which swaras are ornamented and how",
     "difficulty": "intermediate",
     "words": ("gamaka", "gamakas", "ornament", "ornamentation", "kampita",
               "oscillation", "technique")},
    {"key": "jeeva", "title": "{raaga}: jeeva and nyasa swaras",
     "goal": "the life-giving swaras and where a phrase comes to rest",
     "difficulty": "intermediate",
     "words": ("jeeva", "nyasa", "resting", "cadence", "important", "resolve")},
    {"key": "alapana", "title": "{raaga}: alapana shape",
     "goal": "how a free-rhythm exposition of the raaga unfolds",
     "difficulty": "advanced",
     "words": ("alapana", "alapanai", "improvisation", "raga alapana",
               "exposition", "free")},
    {"key": "varisai", "title": "{raaga}: swara exercises",
     "goal": "ascending and descending drills across the raaga's ladder",
     "difficulty": "beginner",
     "words": ("varisai", "sarali", "janta", "exercise", "exercises", "drill",
               "practice", "beginner", "swara")},
    {"key": "mood", "title": "{raaga}: mood and usage",
     "goal": "the feeling the raaga carries and where it is used",
     "difficulty": "beginner",
     "words": ("mood", "feel", "emotion", "rasa", "usage", "when", "bhava")},
    {"key": "tempo", "title": "{raaga}: tempo and pacing",
     "goal": "the tempo range the raaga is usually taken at",
     "difficulty": "beginner",
     "words": ("tempo", "speed", "kala", "pace", "tala", "rhythm", "laya")},
    {"key": "avoid", "title": "{raaga}: phrases to avoid",
     "goal": "the moves that take the raaga somewhere it does not belong",
     "difficulty": "intermediate",
     "words": ("avoid", "avoided", "mistake", "mistakes", "wrong", "common",
               "forbidden", "varjya")},
    {"key": "comparison", "title": "{raaga}: told apart from its neighbours",
     "goal": "what distinguishes it from raagas built on the same swaras",
     "difficulty": "advanced",
     "words": ("compare", "comparison", "difference", "versus", "similar",
               "confuse", "distinguish", "allied")},
    {"key": "structure", "title": "{raaga}: composition structure",
     "goal": "how a composition in the raaga is laid out section by section",
     "difficulty": "advanced",
     "words": ("composition", "structure", "sangati", "sangathi", "pallavi",
               "charanam", "kriti", "varnam", "form")},
)


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if t and t not in STOP_WORDS]


def _transliteration_key(name: str) -> str:
    """Collapse the ways one raaga name gets spelled in Roman letters.

    "Kamboji", "Kambhoji" and "Kambodhi" are one raaga.  Aspirates, doubled
    vowels and the v/w and i/y pairs are exactly where transliterations
    disagree, so they are normalised away before comparison.  This is a
    deliberately blunt instrument: it is used only to *match* a name the
    creator typed against the library, never to rename anything.
    """
    text = (name or "").lower()
    text = re.sub(r"[^a-z]", "", text)
    text = text.replace("bh", "b").replace("dh", "d").replace("th", "t")
    text = text.replace("gh", "g").replace("kh", "k").replace("ph", "p")
    text = text.replace("jh", "j").replace("chh", "ch").replace("sh", "s")
    text = text.replace("w", "v").replace("y", "i")
    text = re.sub(r"(.)\1+", r"\1", text)          # doubled letters
    text = re.sub(r"[aeiou]+", lambda m: m.group(0)[0], text)
    return text


def match_raaga(phrase: str, raagas: RaagaLibrary) -> Optional[Raaga]:
    """Which raaga, if any, is this phrase about?"""
    words = [w for w in re.split(r"[^A-Za-z]+", phrase or "") if len(w) > 2]
    if not words:
        return None
    keys = {_transliteration_key(w): w for w in words}
    for raaga in raagas.all():
        names = [raaga.name] + list(raaga.aliases)
        for name in names:
            key = _transliteration_key(name)
            if key in keys:
                return raaga
            # also allow the typed word to be a prefix of a longer spelling
            for typed in keys:
                if len(typed) >= 5 and (key.startswith(typed)
                                        or typed.startswith(key)):
                    return raaga
    return None


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
class SearchProvider:
    """One place to look. Subclasses return normalised sources."""

    name = "provider"
    #: Providers that can hand over content rank above ones that cannot.
    yields_content = False

    def available(self) -> bool:
        return True

    def search(self, query: SearchQuery, raaga: Optional[Raaga],
               limit: int) -> List[LearningSource]:
        raise NotImplementedError


class ReferenceExerciseProvider(SearchProvider):
    """Lessons the agent can build, perform and hear for itself.

    This is the reason the Training tab works on a machine with no network and
    no material of the creator's own: a student practising a scale to hear it
    is learning from something genuinely available to them.
    """

    name = "exercises"
    yields_content = True

    def __init__(self, raagas: RaagaLibrary) -> None:
        self.raagas = raagas

    def search(self, query: SearchQuery, raaga: Optional[Raaga],
               limit: int) -> List[LearningSource]:
        targets = [raaga] if raaga is not None else list(self.raagas.all())[:3]
        tokens = set(tokenize(query.phrase))
        out: List[LearningSource] = []
        for target in targets:
            if target is None:
                continue
            for topic in EXERCISE_TOPICS:
                if query.difficulty and topic["difficulty"] != query.difficulty:
                    continue
                title = topic["title"].format(raaga=target.name)
                out.append(LearningSource(
                    source_type="exercise",
                    title=title,
                    url=f"raaga-exercise://{target.name}/{topic['key']}",
                    author="the agent's own structural library",
                    description=(f"A rendered exercise the agent performs and "
                                 f"then analyses: {topic['goal']}."),
                    duration=30.0,
                    language="swara notation",
                    accessibility_status=Accessibility.ACCESSIBLE,
                    provider=self.name,
                    metadata={"raaga": target.name, "topic": topic["key"],
                              "goal": topic["goal"],
                              "difficulty": topic["difficulty"],
                              "topic_words": list(topic["words"])}))
                if len(out) >= limit * 2:
                    break
        # Prefer the topics whose own words the creator actually used.
        def topical(source: LearningSource) -> int:
            words = set(source.metadata.get("topic_words", []))
            return len(words & tokens)
        out.sort(key=topical, reverse=True)
        return out


class LocalLibraryProvider(SearchProvider):
    """Recordings the creator put in their own learning folder."""

    name = "library"
    yields_content = True

    def __init__(self, folder: Optional[Path],
                 raagas: Optional[RaagaLibrary] = None) -> None:
        self.folder = Path(folder) if folder else None
        self.raagas = raagas

    def available(self) -> bool:
        return self.folder is not None and self.folder.exists()

    def search(self, query: SearchQuery, raaga: Optional[Raaga],
               limit: int) -> List[LearningSource]:
        if not self.available():
            return []
        tokens = set(tokenize(query.phrase))
        wanted = {raaga.name.lower()} | {a.lower() for a in raaga.aliases} \
            if raaga is not None else set()
        out: List[LearningSource] = []
        for path in sorted(self.folder.rglob("*")):
            if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
                continue
            blob = f"{path.parent.name} {path.stem}".lower()
            if wanted and not any(name in blob for name in wanted):
                continue
            if not wanted and tokens and not (set(tokenize(blob)) & tokens):
                continue
            named = raaga.name if raaga is not None else self._raaga_in(blob)
            out.append(LearningSource(
                source_type="local_file", title=path.stem, url=path.as_uri(),
                author="your own learning folder",
                description=f"Audio you supplied, in {path.parent.name}.",
                accessibility_status=Accessibility.ACCESSIBLE,
                provider=self.name, local_path=str(path),
                metadata={"raaga": named, "rights": "user-supplied"}))
            if len(out) >= limit:
                break
        return out

    def _raaga_in(self, text: str) -> str:
        """Which raaga a file names, when the search phrase did not say."""
        if self.raagas is None:
            return ""
        found = match_raaga(text, self.raagas)
        return found.name if found is not None else ""


class WebLeadProvider(SearchProvider):
    """Off by default. Records where material is said to be; fetches nothing.

    Specification section 4 forbids circumventing access controls, and section
    20 rule 9 requires inaccessible content to be reported honestly.  A lead
    satisfies both: it tells the creator where to look and asks them to supply
    anything they are entitled to use, and it never pretends the lesson was
    learned.
    """

    name = "web"
    yields_content = False

    def __init__(self, enabled: bool = False,
                 finder: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None
                 ) -> None:
        self.enabled = enabled
        self.finder = finder

    def available(self) -> bool:
        return bool(self.enabled)

    def search(self, query: SearchQuery, raaga: Optional[Raaga],
               limit: int) -> List[LearningSource]:
        if not self.available() or self.finder is None:
            return []
        try:
            found = self.finder(query.phrase, limit) or []
        except Exception as exc:  # noqa: BLE001 - a lead is never worth a crash
            log.warning("web leads unavailable: %s", exc)
            return []
        out: List[LearningSource] = []
        for item in found[:limit]:
            out.append(LearningSource(
                source_type="lead",
                title=str(item.get("title", ""))[:200],
                url=str(item.get("url", "")),
                author=str(item.get("author", "")),
                description=(str(item.get("description", ""))[:500]
                             + "  Lead only - nothing has been fetched. "
                               "Supply the file yourself if you are entitled "
                               "to use it."),
                language=str(item.get("language", "")),
                accessibility_status=Accessibility.METADATA_ONLY,
                provider=self.name,
                metadata={"rights": "external-unverified"}))
        return out


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------
class LearningSourceSearchService:
    """Section 5.  Search, normalise, rank, de-duplicate, return."""

    def __init__(self, store: TrainingStore, raagas: RaagaLibrary,
                 settings: Optional[Settings] = None,
                 providers: Optional[Sequence[SearchProvider]] = None,
                 web_finder=None) -> None:
        self.store = store
        self.raagas = raagas
        self.settings = settings or Settings.load()
        corpus = getattr(self.settings, "learning_corpus_dir", "")
        self.providers: List[SearchProvider] = list(providers) if providers \
            else [
                LocalLibraryProvider(Path(corpus) if corpus else None,
                                     raagas),
                ReferenceExerciseProvider(raagas),
                WebLeadProvider(
                    bool(getattr(self.settings, "training_allow_web", False)),
                    web_finder),
            ]

    # ------------------------------------------------------------------
    def search(self, query: SearchQuery) -> List[LearningSource]:
        limit = max(1, int(query.max_results or 10))
        raaga = match_raaga(query.phrase, self.raagas)
        if raaga is not None:
            log.info("search '%s' reads as raaga %s", query.phrase, raaga.name)

        gathered: List[LearningSource] = []
        for provider in self.providers:
            if query.source_filter and provider.name != query.source_filter:
                continue
            if not provider.available():
                continue
            try:
                found = provider.search(query, raaga, limit)
            except Exception as exc:  # noqa: BLE001
                log.warning("provider %s failed: %s", provider.name, exc)
                continue
            for source in found:
                source.relevance_score = self._score(source, query, raaga,
                                                     provider)
                gathered.append(source)

        kept = self._deduplicate(gathered)
        kept = [s for s in kept if self._passes_filters(s, query)]
        for source in kept:
            self._mark_if_already_learned(source)
        kept.sort(key=lambda s: s.relevance_score, reverse=True)
        kept = kept[:limit]

        search_id = self.store.record_search(query, len(kept))
        for source in kept:
            source.search_id = search_id
            self.store.save_candidate(source)
        log.info("search '%s' returned %d candidate(s)", query.phrase,
                 len(kept))
        return kept

    # ------------------------------------------------------------------
    def _score(self, source: LearningSource, query: SearchQuery,
               raaga: Optional[Raaga], provider: SearchProvider) -> float:
        """Rank for usefulness *for learning*, not for textual similarity.

        A perfectly-titled source we may not open teaches nobody anything, so
        being analysable counts for more here than another matching word.
        """
        wanted = set(tokenize(query.phrase)) | {
            t for k in query.include_keywords for t in tokenize(k)}
        haystack = set(tokenize(f"{source.title} {source.description} "
                                f"{source.author}"))
        overlap = len(wanted & haystack) / max(1, len(wanted)) if wanted else 0.0

        topical = set(source.metadata.get("topic_words", []))
        topic_hit = 1.0 if (topical & wanted) else 0.0

        raaga_hit = 0.0
        if raaga is not None:
            named = str(source.metadata.get("raaga", "")).lower()
            if named == raaga.name.lower():
                raaga_hit = 1.0

        difficulty_hit = 0.0
        if query.difficulty:
            difficulty_hit = 1.0 if str(
                source.metadata.get("difficulty", "")) == query.difficulty \
                else -0.5

        analysable = 1.0 if source.can_be_analysed else 0.0
        content = 0.5 if provider.yields_content else 0.0

        score = (0.28 * overlap + 0.16 * topic_hit + 0.20 * raaga_hit
                 + 0.24 * analysable + 0.12 * content
                 + 0.10 * difficulty_hit)
        return round(max(0.0, min(1.0, score)), 4)

    def _passes_filters(self, source: LearningSource,
                        query: SearchQuery) -> bool:
        text = f"{source.title} {source.description}".lower()
        for word in query.exclude_keywords:
            if word.strip() and word.strip().lower() in text:
                return False
        if query.content_type and source.source_type != query.content_type:
            return False
        if query.language and source.language and \
                query.language.lower() not in source.language.lower():
            return False
        if query.duration_preference and source.duration > 0:
            minutes = source.duration / 60.0
            bounds = {"short": (0, 8), "medium": (8, 25), "long": (25, 1e9)}
            low, high = bounds.get(query.duration_preference, (0, 1e9))
            if not low <= minutes < high:
                return False
        return True

    @staticmethod
    def _deduplicate(sources: Sequence[LearningSource]
                     ) -> List[LearningSource]:
        """One entry per lesson, keeping the best-scoring way of reaching it."""
        best: Dict[str, LearningSource] = {}
        for source in sources:
            key = identity_of(source.url, source.title)
            current = best.get(key)
            if current is None or source.relevance_score > current.relevance_score:
                best[key] = source
        return list(best.values())

    def _mark_if_already_learned(self, source: LearningSource) -> None:
        """Section 10: say so rather than quietly learning it twice."""
        previous = self.store.completed_run_for(source)
        if previous is not None:
            source.previously_learned = True
            source.metadata = dict(source.metadata,
                                   previous_run_id=previous.run_id)
