"""TEST G from the canonical specification (v0.3 section 63), through the
controller the creator actually uses.

    Select Keeravani and request a short prelude.
    PASS: multi-note output, not a single sustained note, minimum raga
    validity, playable result.

Section 8 adds that Generate Tune must never return a trivial single
sustained note when meaningful melodic generation is requested, and section
54 that a silent or single-note artifact must never be treated as valid
output.  The agent's own practice path is covered by
``tests/integration/test_agent_acceptance.py``; this test exercises
``AppController.generate_tune`` and ``render("tune")`` - the buttons.
"""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.music.validator import validate

pytestmark = pytest.mark.regression


def test_g_generate_tune_for_keeravani_is_real_playable_music(app, settle):
    app.update_brief(situation="a quiet dawn", mood="devotional",
                     feel="calm and still", duration_target=30.0)
    app.select_raaga("Keeravani")
    app.generate_tune(seed=5)
    settle()

    melody = app.project.melody()
    assert melody is not None, "Generate Tune produced no melody version"
    assert melody.raaga == "Keeravani"
    notes = melody.notes
    assert len(notes) >= 8, "a 30-second tune must be more than a few notes"
    assert len({round(n.midi) for n in notes}) >= 4, \
        "a tune must move, not sit on one pitch"
    span = max(n.start + n.duration for n in notes)
    assert span > 10.0, "requested length must be respected"
    assert melody.sections, "a tune has a beginning, development and ending"

    raaga = app.raagas.require("Keeravani")
    assert validate(melody, raaga).stats["out_of_raaga"] == 0

    app.render("tune", autoplay=False)
    settle()
    rendered = app._renders.get("tune")
    assert rendered is not None, "the tune must be playable"
    assert rendered.audio.size > rendered.sample_rate, \
        "rendered audio is shorter than a second"
    assert float(np.max(np.abs(rendered.audio))) > 0.01, "rendered audio is silent"
