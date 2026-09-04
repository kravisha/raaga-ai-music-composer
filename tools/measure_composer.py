"""Measure the composer's output for Keeravani (docs/PLAN_learning_loop.md, item 3).

Builds a MusicAgent, teaches it Keeravani through the prayogas unit the same
way ``tests/regression/test_agent_regressions.py``'s ``_teach_through_prayogas``
does (real research, real practice, no mocks), takes the learned view of
Keeravani, generates a tune for each seed the same way ``App.generate_tune``
does, evaluates it with the agent's own evaluator, and checks it against the
agent's own originality index.  Prints a markdown table of the mean of every
evaluator dimension plus a few summary rows.

Three tables are worth comparing side by side, one run each:

* the pre-item-3 baseline (run against a checkout of ``main`` before any of
  this work landed - this script did not exist yet, so that run used a copy
  of it dropped into that checkout);
* this tree with ``--no-idiom`` - fragment quoting and the scale-run
  exclusion (``agent/learned.py``, ``_is_scale_run``/``SCALE_RUN_MIN``) are
  both still in effect even with the idiom stripped off, since both sit
  upstream of the idiom in ``learned_raaga`` and in ``_phrase_tokens``'s
  quoting branch - only the direction/step/cadence draws that consult
  ``raaga.idiom`` are turned off;
* this tree with the idiom left on (the default), so all three of fragment
  quoting, scale-run exclusion and the idiom itself are in effect.

Run::

    .venv\\Scripts\\python.exe tools\\measure_composer.py [--seeds 20] [--no-idiom]

``--no-idiom`` strips the idiom from the learned raaga view
(``dataclasses.replace(raaga, idiom=None)``) so the idiom-aware and
idiom-free paths can be compared against the same taught knowledge.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before raagacomposer.core.settings is first imported - the same
# throwaway-home dance tests/conftest.py does, so this script never touches a
# real config directory or spends real money on a provider.
_HOME = Path(tempfile.mkdtemp(prefix="raaga-measure-home-"))
os.environ["RAAGA_COMPOSER_HOME"] = str(_HOME)
os.environ["RAAGA_SECRET_BACKEND"] = "file"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("ANTHROPIC_API_KEY", None)
(_HOME / "settings.json").write_text(json.dumps({
    "llm_provider": "off",
    "llm_routing": "off",
    "projects_dir": str(_HOME / "projects"),
    "recent_projects": [],
}, indent=2), encoding="utf-8")

from raagacomposer.agent.evaluator import DIMENSIONS               # noqa: E402
from raagacomposer.agent.music_agent import MusicAgent             # noqa: E402
from raagacomposer.agent.originality import check as check_originality  # noqa: E402
from raagacomposer.core.settings import Settings                   # noqa: E402
from raagacomposer.music import melody as melody_engine            # noqa: E402
from raagacomposer.music.melody import MelodyOptions               # noqa: E402
from raagacomposer.music.structure import plan_sections            # noqa: E402

RAAGA_NAME = "Keeravani"
PRAYOGA_UNIT = "b06.prayogas:Keeravani"


def _settings(tmp_dir: Path) -> Settings:
    s = Settings.load()
    s.projects_dir = str(tmp_dir / "projects")
    s.autosave_seconds = 5
    s.stt_provider = "none"
    # Each run gets its own memory, same as the test fixture.
    s.knowledge_db = str(tmp_dir / "knowledge.db")
    s.learning_corpus_dir = ""
    s.learning_allow_web = False
    s.learning_autostart = False
    return s


def _teach_through_prayogas(agent: MusicAgent, max_steps: int = 40) -> None:
    """The same path tests/regression/test_agent_regressions.py's
    ``_teach_through_prayogas`` takes: real research and real practice, no
    mocks, until the agent has heard Keeravani's characteristic phrases."""
    for _ in range(max_steps):
        if agent.repo.progress(PRAYOGA_UNIT).status == "passed":
            break
        step = agent.learn_step()
        if step.action == "idle":
            break


def measure(seeds: int, strip_idiom: bool) -> dict:
    tmp_dir = Path(tempfile.mkdtemp(prefix="raaga-measure-"))
    settings = _settings(tmp_dir)
    agent = MusicAgent(settings)
    try:
        _teach_through_prayogas(agent)

        raaga, completeness = agent.raaga_for_composition(RAAGA_NAME)
        if raaga is None:
            raise RuntimeError("agent has no view of Keeravani after teaching")
        has_idiom = getattr(raaga, "idiom", None) is not None
        if strip_idiom:
            raaga = dataclasses.replace(raaga, idiom=None)

        evaluator = agent.evaluator(RAAGA_NAME)
        index = agent.phrase_index(RAAGA_NAME)
        learned_phrases = agent.phrase_bank(RAAGA_NAME)

        totals = {d: 0.0 for d in DIMENSIONS}
        overall_total = 0.0
        note_count_total = 0
        first_try_fail = 0
        for seed in range(1, seeds + 1):
            opts = MelodyOptions(seed=seed, tempo_bpm=72, duration_target=45.0)
            sections = plan_sections(opts.duration_target, opts.tempo_bpm,
                                     opts.beats_per_cycle, opts.song_type)
            melody = melody_engine.generate(raaga, opts, sections)
            notes = melody.notes
            evaluation = evaluator.evaluate(
                notes, raaga, tonic_midi=opts.tonic_midi,
                tempo_bpm=opts.tempo_bpm, expected_seconds=45.0,
                learned_phrases=learned_phrases)
            for d in DIMENSIONS:
                totals[d] += evaluation.scores.get(d, 0.0)
            overall_total += evaluation.overall()
            note_count_total += len(notes)

            swaras = [n.swara for n in notes]
            report = check_originality(swaras, index)
            if not report.is_original:
                first_try_fail += 1

        return {
            "dimensions": {d: totals[d] / seeds for d in DIMENSIONS},
            "overall": overall_total / seeds,
            "first_try_originality_fail": first_try_fail,
            "seeds": seeds,
            "mean_notes": note_count_total / seeds,
            "has_idiom": has_idiom,
            "idiom_used": has_idiom and not strip_idiom,
            "phrases_heard": len(learned_phrases),
        }
    finally:
        agent.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_table(stats: dict, title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append(f"Idiom available from learning: **{stats['has_idiom']}**; "
                f"used for this run: **{stats['idiom_used']}**  \n"
                f"Phrases heard through the prayogas unit: "
                f"{stats['phrases_heard']}")
    lines.append("")
    lines.append("| Dimension | Mean over seeds |")
    lines.append("|---|---|")
    for d in DIMENSIONS:
        lines.append(f"| {d} | {stats['dimensions'][d]:.3f} |")
    lines.append(f"| **overall** | **{stats['overall']:.3f}** |")
    lines.append("")
    lines.append(f"- Seeds failing originality on the first try (would fire "
                f"`generate_tune`'s rewrite loop): "
                f"{stats['first_try_originality_fail']} / {stats['seeds']}")
    lines.append(f"- Mean notes per tune: {stats['mean_notes']:.1f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20,
                        help="number of seeds 1..N to average over")
    parser.add_argument("--no-idiom", action="store_true",
                        help="strip the idiom from the learned raaga view "
                             "before generating, for comparison")
    args = parser.parse_args()

    stats = measure(args.seeds, args.no_idiom)
    mode = "idiom stripped" if args.no_idiom else "as generated"
    title = f"Keeravani composer measurement ({mode}, seeds 1..{args.seeds})"
    print(render_table(stats, title))


if __name__ == "__main__":
    main()
