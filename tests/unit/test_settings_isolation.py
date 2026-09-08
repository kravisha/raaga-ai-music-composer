"""One test cannot change another's settings, and a diagnostic cannot
silently open a real home.

Found the hard way on 2026-09-08, twice in one afternoon: saving a project
calls Settings.remember_project, which writes the *whole* settings object
to settings.json, and switching the workspace does the same.  With every
test reading and writing the one settings.json in the shared test home, a
"stop" policy set by one test and a LEARN workspace switched by another
each reached tests that ran later, and cost a full-suite failure apiece.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

import pytest

from raagacomposer.core.settings import Settings, config_dir

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _shared_settings_file() -> Path:
    return config_dir() / "settings.json"


def test_saving_settings_writes_the_test_s_own_file_not_the_shared_home():
    shared = _shared_settings_file()
    before = shared.read_text(encoding="utf-8") if shared.exists() else ""
    sentinel = f"leak-{uuid.uuid4().hex}"
    settings = Settings.load()
    settings.extra["sentinel"] = sentinel
    settings.save()
    assert Settings.path() != shared, \
        "this test's settings file is the shared home's settings file"
    after = shared.read_text(encoding="utf-8") if shared.exists() else ""
    assert sentinel not in after and after == before, \
        "a settings save reached the shared test home"
    assert Settings.load().extra.get("sentinel") == sentinel, \
        "the save did not persist where the next load reads"


def test_a_project_save_leaves_the_shared_home_settings_untouched(app):
    shared = _shared_settings_file()
    digest_before = hashlib.sha256(shared.read_bytes()).hexdigest()
    app.new_project("Registered somewhere", write=False)
    assert app.save() is not None
    assert hashlib.sha256(shared.read_bytes()).hexdigest() == digest_before, \
        "registering a project rewrote the shared settings.json"
    # The registration did happen - into this test's own settings file.
    assert str(app.project_dir) in Settings.load().recent_projects


def test_a_workspace_switch_stays_in_its_own_test():
    shared = _shared_settings_file()
    settings = Settings.load()
    settings.extra["workspace"] = "LEARN"
    settings.save()
    assert '"LEARN"' not in shared.read_text(encoding="utf-8")
    assert Settings.load().extra.get("workspace") == "LEARN"


def test_the_measuring_tool_takes_a_throwaway_home_before_importing_the_app():
    """tools/measure_composer.py constructs the whole agent; it must set
    RAAGA_COMPOSER_HOME before the first application import, as conftest
    does, or it opens the creator's real databases (2026-09-07)."""
    source = (ROOT / "tools" / "measure_composer.py").read_text(encoding="utf-8")
    home_set = re.search(r'os\.environ\["RAAGA_COMPOSER_HOME"\]\s*=', source)
    first_import = re.search(r"^\s*(from|import) raagacomposer", source, re.M)
    assert home_set and first_import, "shape of the tool changed; re-check its guard"
    assert home_set.start() < first_import.start(), \
        "measure_composer imports the application before setting a throwaway home"
