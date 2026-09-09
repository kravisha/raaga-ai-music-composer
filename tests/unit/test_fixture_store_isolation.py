"""Native fixtures must not carry training or Knowledge Base data into a test."""
from pathlib import Path

import pytest

from raagacomposer.app import AppController
from raagacomposer.core.settings import Settings
from raagacomposer.training.models import KnowledgeEntry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("attempt", range(2))
def test_app_training_store_starts_clean_on_each_fixture(app, attempt):
    marker = "native-training-fixture-isolation"
    assert not app.training.search_knowledge(keyword=marker)
    app.training.store.add_knowledge(KnowledgeEntry(
        normalized_statement=marker, subject="fixture", source_title="fixture"))


@pytest.mark.parametrize("attempt", range(2))
def test_app_knowledge_base_starts_clean_on_each_fixture(app, attempt):
    marker = "native-kb-fixture-isolation"
    assert app.kb.store.get_meta(marker) is None
    app.kb.store.set_meta(marker, "written by this fixture")


@pytest.fixture(scope="module")
def module_controller():
    # UI modules construct controllers from Settings.load rather than the
    # function-scoped settings fixture. Their default stores need scope too.
    settings = Settings.load()
    settings.stt_provider = "none"
    controller = AppController(settings)
    controller.training.store.add_knowledge(KnowledgeEntry(
        normalized_statement="module-controller-fixture-data",
        subject="fixture", source_title="module fixture"))
    controller.kb.store.set_meta("module-controller-fixture-data", "module")
    try:
        yield controller
    finally:
        controller.close()


def test_direct_settings_controller_does_not_inherit_module_stores(module_controller):
    settings = Settings.load()
    settings.stt_provider = "none"
    controller = AppController(settings)
    try:
        assert controller.training.store.path != module_controller.training.store.path
        assert controller.kb.store.path != module_controller.kb.store.path
        assert controller.agent.repo.path != module_controller.agent.repo.path
        assert controller.settings.factory_db != module_controller.settings.factory_db
        assert not controller.training.search_knowledge(keyword="module-controller-fixture-data")
        assert controller.kb.store.get_meta("module-controller-fixture-data") is None
        assert controller.settings.projects_dir != module_controller.settings.projects_dir
    finally:
        controller.close()


def test_module_defaults_are_beside_its_settings_file(module_controller,
                                                    own_settings_file_for_module):
    directory = own_settings_file_for_module.parent
    settings = module_controller.settings
    for field in ("knowledge_db", "factory_db", "training_db", "knowledge_base_db"):
        assert getattr(settings, field), f"{field} still falls back to the shared home"
        assert Path(getattr(settings, field)).parent == directory
    assert Path(settings.projects_dir).parent == directory
