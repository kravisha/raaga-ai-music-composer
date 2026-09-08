"""A module-scoped fixture that saves settings must not reach the shared
test home either.

The per-test settings file (conftest ``own_settings_file``) is
function-scoped, so a module-scoped fixture's setup and teardown run
outside it - and two of the offscreen window modules save a LEARN
workspace and a projects_dir from exactly there.  In the de0141e suite the
shared file carried both into ``test_a_workspace_switch_stays_in_its_own_test``
(2026-09-08).  Every module gets its own copy now, and every test its own
on top of that.
"""
from pathlib import Path

import pytest

from raagacomposer.core.settings import Settings, config_dir

pytestmark = pytest.mark.unit

SENTINEL = "module-fixture-leak"


@pytest.fixture(scope="module")
def saved_by_the_module():
    settings = Settings.load()
    settings.extra["workspace"] = "LEARN"
    settings.extra["sentinel"] = SENTINEL
    settings.save()
    yield settings
    settings.extra["sentinel"] = SENTINEL + "-teardown"
    settings.save()


def test_a_module_fixture_s_save_stays_out_of_the_shared_home(saved_by_the_module):
    shared = (config_dir() / "settings.json").read_text(encoding="utf-8")
    assert SENTINEL not in shared and '"LEARN"' not in shared
    # And the test's own file starts from the template, not from whatever
    # the module fixture wrote.
    assert Settings.load().extra.get("workspace") != "LEARN"


def test_the_module_fixture_still_has_what_it_saved(saved_by_the_module):
    assert saved_by_the_module.extra["sentinel"] == SENTINEL
    assert saved_by_the_module.extra["workspace"] == "LEARN"
    assert SENTINEL not in (config_dir() / "settings.json").read_text(encoding="utf-8")
