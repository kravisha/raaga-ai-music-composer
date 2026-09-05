"""Find knowledge derived by code that no longer exists, and re-derive it.

The agent learns by listening, and what it hears depends on the code that
did the listening.  In one session the extraction changed four times - swara
naming went free for outside material, notes within a gamaka's reach were
given the raaga's own name, octave slips were folded back into the phrase,
and phrases still leaping were refused.  Each of those makes the *same*
recording yield different phrases.

So a phrase learned last week is not knowledge about a raaga; it is
knowledge about a raaga *as heard by a particular version of the ears*.
When the ears change, it has to be re-derived or it is quietly wrong - and
nothing about a stale phrase looks wrong, which is what makes this
dangerous.

Every source row records the ``extraction_version`` that produced it.  This
compares that against the code and reports, or rebuilds:

    python tools/rebuild_knowledge.py                 # what is stale
    python tools/rebuild_knowledge.py --rebuild       # re-derive it

Rebuilding deletes the phrases and facts of stale sources and ingests their
audio again from the training manifest.  The audio is the thing of record;
what was derived from it is not.  Nothing is deleted until the audio has
been located, so a rebuild cannot leave you with less than you started.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raagacomposer.agent import analysis                        # noqa: E402
from raagacomposer.agent.knowledge import KnowledgeRepository    # noqa: E402
from raagacomposer.agent.research import (LocalCorpusProvider,   # noqa: E402
                                          ResearchAgent)
from raagacomposer.core.settings import Settings, config_dir     # noqa: E402
from raagacomposer.kb.service import KnowledgeBaseService        # noqa: E402
from raagacomposer.raaga.library import library                  # noqa: E402


def stale_sources(repo: KnowledgeRepository) -> list:
    """Sources whose extraction version is not the one in the code."""
    current = analysis.ANALYSIS_VERSION
    out = []
    for source in repo.sources():
        version = (source.extraction_version or "").split("+")[0]
        if version and version != current:
            out.append((source, version))
    return out


def report(repo: KnowledgeRepository) -> int:
    current = analysis.ANALYSIS_VERSION
    every = repo.sources()
    stale = stale_sources(repo)
    print(f"the ears are at {current!r}")
    print(f"{len(every)} source(s) recorded, {len(stale)} stale\n")
    for source, version in stale:
        phrases = [p for p in repo.phrases(limit=5000)
                   if p.source_id == source.id]
        print(f"  {version:22s} {len(phrases):4d} phrase(s)  "
              f"{source.title[:44]}")
    if not stale:
        print("  everything was derived by the current code")
    return 0


def backup(paths) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in paths:
        if path.exists():
            target = path.with_name(f"{path.stem}.before-rebuild-{stamp}{path.suffix}")
            shutil.copy2(path, target)
            print(f"  backed up {path.name} -> {target.name}")


def rebuild(settings: Settings, repo: KnowledgeRepository, kb) -> int:
    stale = stale_sources(repo)
    if not stale:
        print("nothing to rebuild")
        return 0

    # Locate the audio before deleting anything.  A rebuild that cannot find
    # the recordings must not be a rebuild that lost the phrases.
    agent = ResearchAgent(repo, library(), settings, kb=kb)
    corpus = next((p for p in agent.providers
                   if isinstance(p, LocalCorpusProvider)), None)
    if corpus is None or not corpus.folder or not corpus.folder.exists():
        print("the learning folder is not available; refusing to delete "
              "knowledge that could not be re-derived")
        return 1

    raagas = sorted({s.raaga for s, _ in stale if s.raaga})
    reachable = {}
    for name in raagas:
        raaga = library().get(name)
        if raaga is None:
            continue
        reachable[name] = corpus.find(raaga, "phrases", 50)
    missing = [n for n in raagas if not reachable.get(n)]
    if missing:
        print(f"no audio found for {', '.join(missing)} - refusing to delete "
              f"knowledge that could not be re-derived")
        return 1

    print(f"\nre-deriving {len(raagas)} raaga(s) from audio: "
          f"{', '.join(raagas)}")
    for source, _ in stale:
        removed = repo.forget_source(source.id)
        print(f"  forgot {removed} item(s) from {source.title[:40]}")

    learned = 0
    for name, candidates in reachable.items():
        for candidate in candidates:
            result = agent.ingest(candidate)
            learned += result.phrases_learned
            print(f"  {name}: {result.summary()}")
    print(f"\nre-derived {learned} phrase(s) at {analysis.ANALYSIS_VERSION!r}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="re-derive stale knowledge, not just report it")
    args = parser.parse_args(argv)

    settings = Settings.load()
    home = config_dir()
    repo = KnowledgeRepository(str(home / "knowledge.db"))
    kb = KnowledgeBaseService.initialize_if_needed(home / "knowledge_base.db")
    try:
        if not args.rebuild:
            return report(repo)
        print("BACKUPS")
        backup([home / "knowledge.db", home / "knowledge_base.db"])
        report(repo)
        return rebuild(settings, repo, kb)
    finally:
        repo.close()
        kb.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
