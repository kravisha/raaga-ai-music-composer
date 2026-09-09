"""Trial 1 — one short complete Tamil phrase, two lines, with a rest between.

Contract `46e5ca2` (v2) §1.5: short and complete, in the 10–20 s range, one
phrase to begin with. The rest between the lines is deliberate — it is the
thing the reviewed implementation sang through, so the first real trial should
contain one.

Lyrics and melody are authored for this experiment. Not from any released song,
and not a transcription of one.

Run:
    python -m trials.nilavaai_two_lines           # guide only, no model
    python -m trials.nilavaai_two_lines --convert # the full milestone
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_prototype import pipeline, tune as tune_mod          # noqa: E402
from voice_prototype.tune import Event                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRIAL_CVVC = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Tamil-CVVC-Trial-20260908")
QUALITY = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Vocal-Quality-20260909")

LYRICS = "நிலவாய் நீ வந்தால் இரவும் பகலாகும்"
MEANING = "if you came as the moon, the night would become day"

#: One akshara per slot, with the word starts marked. The word-initial flag is
#: what makes the plosive voicing right - `pa` at the front of a word, `ga`
#: inside one - and that is the difference between two words, not a nicety.
SLOTS = [
    {"draft_articulation": "நி", "word_initial": True},
    {"draft_articulation": "ல", "word_initial": False},
    {"draft_articulation": "வாய்", "word_initial": False},
    {"draft_articulation": "நீ", "word_initial": True},
    {"draft_articulation": "வந்", "word_initial": True},
    {"draft_articulation": "தால்", "word_initial": False},
    {"draft_articulation": "இ", "word_initial": True},
    {"draft_articulation": "ர", "word_initial": False},
    {"draft_articulation": "வும்", "word_initial": False},
    {"draft_articulation": "ப", "word_initial": True},
    {"draft_articulation": "க", "word_initial": False},
    {"draft_articulation": "லா", "word_initial": False},
    {"draft_articulation": "கும்", "word_initial": False},
]

#: Direct melody controls - the preferred path. Fractional targets and explicit
#: bends, so the ornament detail is written rather than approximated afterwards.
#: A rest of 0.9 s separates the two lines.
EVENTS = [
    # line 1
    Event(0.00, 0.80, midi=67.0, bend=[(0.0, -25.0), (0.12, 0.0)]),
    Event(0.80, 0.50, midi=69.0),
    Event(1.30, 1.10, midi=71.0, bend=[(0.0, 0.0), (0.45, 35.0), (0.9, 0.0)]),
    Event(2.40, 0.90, midi=72.0),
    Event(3.30, 0.60, midi=70.5),
    Event(3.90, 1.50, midi=67.0, bend=[(0.0, 20.0), (0.6, -15.0), (1.4, 0.0)]),
    Event(5.40, 0.90, midi=None),                       # the rest
    # line 2
    Event(6.30, 0.70, midi=69.0),
    Event(7.00, 0.50, midi=71.0),
    Event(7.50, 1.00, midi=72.0, bend=[(0.0, 0.0), (0.5, 40.0), (1.0, 0.0)]),
    Event(8.50, 0.70, midi=74.0),
    Event(9.20, 0.60, midi=72.0),
    Event(9.80, 0.80, midi=70.5),
    Event(10.60, 1.90, midi=67.0, bend=[(0.0, 30.0), (0.8, -20.0), (1.8, 0.0)]),
]

#: The existing authorized reference, read where it lies, with the limits its
#: own record states. Not padded, and not relabelled.
REFERENCE = pipeline.Reference(
    path=QUALITY / "artist-reference-20260909" / "spb-credited-reference-8s.wav",
    sha256="7ba960f2cf3dcbd39166c1209d762e4b5ba9d6f07f37adee22e0c645d5161920",
    use_scope=("User-authorized local private trial with an existing download; "
               "no public distribution and no rights-clearance claim."),
    limits=[
        "Synthetic reference-conditioned output, not an authentic SPB performance.",
        "The source is a song mix with accompaniment, not an isolated vocal stem; "
        "no source separation and no speaker recognition were used.",
        "One 8-second excerpt of one song does not demonstrate versatility.",
        "8 seconds is what exists and is used as-is; it is not padded to reach a "
        "nominal duration.",
    ],
    parent_source=("Kaadhalin Deepam, existing local download; see "
                   "artist-reference-20260909/reference-extraction.json"),
    interval_s=[40.0, 48.0],
    is_isolated_vocal=False,
)

BANK = pipeline.Bank(TRIAL_CVVC / "Hanakoloid.zip",
                     TRIAL_CVVC / "runtimes_win-x64_native_worldline.dll")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    convert = None
    if "--convert" in argv:
        from voice_prototype.convert import SeedVC
        convert = SeedVC(seed=20260909, compared_baseline="spb_conditioned")

    melody = tune_mod.from_events(EVENTS, notes=(
        "Authored for this trial. Direct melody controls, so the curve is the "
        "melody rather than an estimate of it."))
    result = pipeline.sing(
        lyrics=LYRICS, slots=SLOTS, melody=melody, reference=REFERENCE, bank=BANK,
        out_root=ROOT / "renders", convert=convert, decided_by="Claude",
        notes=(f"Trial 1. {MEANING}. Two lines with a 0.9 s rest between them, "
               f"{melody.duration_s:.1f} s total. Lyrics and melody authored for "
               f"this experiment; not from any released song."))
    print(f"render {result['render_id']}  complete={result['complete']}")
    print(f"  guide     {result['guide']}")
    if result["converted"]:
        print(f"  converted {result['converted']}")
    print(f"  record    {result['record']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
