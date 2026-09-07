"""The tests must never resolve to the creator's own configuration home.

On 2026-09-07 a hand-written probe constructed an AppController without
setting RAAGA_COMPOSER_HOME.  It therefore opened the production
knowledge.db and knowledge_base.db while the application had them open, and
both were damaged.  conftest sets the variable for the suite; this asserts
it, so the protection is checked rather than assumed - and so anyone
writing a probe has something to copy.
"""
from __future__ import annotations

import os
from pathlib import Path

from raagacomposer.core.settings import config_dir

#: Where the real thing lives on Windows.  Nothing under test may resolve
#: here, whatever else changes.
PRODUCTION = Path(os.path.expandvars(r"%APPDATA%\RaagaComposer"))


def test_the_test_home_is_not_the_creators_home():
    home = config_dir()
    assert os.environ.get("RAAGA_COMPOSER_HOME"), \
        "RAAGA_COMPOSER_HOME is unset, so this would use the real home"
    assert home == Path(os.environ["RAAGA_COMPOSER_HOME"]), \
        f"config_dir() is {home}, not the test home"


def test_the_test_home_is_not_the_production_directory():
    home = config_dir().resolve()
    try:
        production = PRODUCTION.resolve()
    except OSError:                     # not a Windows layout; nothing to do
        return
    assert home != production, \
        f"the tests would write into the creator's own home: {home}"
    assert production not in home.parents, \
        f"the test home sits inside the creator's own home: {home}"


def test_the_databases_under_test_are_inside_the_test_home():
    """Not just the directory: the files the agent opens have to be there."""
    home = config_dir().resolve()
    for name in ("knowledge.db", "knowledge_base.db", "factory.db"):
        path = (home / name).resolve()
        assert home in path.parents or path.parent == home, \
            f"{name} resolves outside the test home: {path}"
