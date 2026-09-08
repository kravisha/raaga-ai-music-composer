"""Integration: a locked arrangement region survives a rewrite of the tune.

Real controller.  Lock a region, rewrite a section, arrange again: the
region the creator locked comes through untouched, everything else is
rebuilt on the new tune, and nothing errors.
"""
from __future__ import annotations

import time

import pytest

from raagacomposer.core.models import SectionKind

pytestmark = pytest.mark.integration


def _settle(app, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.pump()
        if not app.jobs.active_jobs():
            app.pump()
            if not app.jobs.active_jobs():
                return
        time.sleep(0.02)
    raise TimeoutError([j.description for j in app.jobs.active_jobs()])


def _a_tune(app, title):
    app.new_project(title, write=False)
    app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                     situation="A hopeful reunion after a long separation",
                     notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                           "Charanam and Ending.")
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=37)
    _settle(app)
    assert app.project.melody() is not None, app.status_text


def _signature(region):
    return (round(region.start, 3), round(region.end, 3), region.seed,
            [(n.start, n.midi) for n in region.notes])


def test_a_locked_region_survives_a_section_rewrite_and_a_new_arrangement(app):
    _a_tune(app, "Locked region through a rewrite")
    app.auto_arrange()
    _settle(app)
    first = app.project.arrangement()
    assert first is not None, app.status_text
    from raagacomposer.music import arrangement as arranger
    lead = next(t for t in first.tracks if t.role == "lead")
    interlude = next(s for s in app.project.melody().sections
                     if s.kind is SectionKind.INTERLUDE)
    region = next(r for r in lead.regions if r.overlaps(interlude.start, interlude.end))
    arranger.set_region_lock(first, lead.id, region.id, True)
    locked = [_signature(r) for t in first.tracks for r in t.regions if r.locked]
    assert len(locked) == 1
    pallavi = next(s for s in app.project.melody().sections
                   if s.kind is SectionKind.PALLAVI)
    v1 = app.project.melody().version

    app.regenerate_tune_section(pallavi.id)
    _settle(app)
    assert app.project.melody().version == v1 + 1, app.status_text
    app.auto_arrange()
    _settle(app)
    second = app.project.arrangement()
    assert second is not None and second.version == first.version + 1, app.status_text
    assert "failed" not in app.status_text.lower(), app.status_text
    still = [_signature(r) for t in second.tracks for r in t.regions if r.locked]
    assert still == locked, "a locked region did not survive the new arrangement"
    lead2 = next(t for t in second.tracks if t.role == "lead")
    assert [r for r in lead2.regions if not r.locked], "the lead's other sections were not rebuilt"
    assert len(lead2.regions) == len(lead.regions)
    pad2 = next(t for t in second.tracks if t.role == "pad")
    assert pad2.regions and not any(r.locked for r in pad2.regions)
