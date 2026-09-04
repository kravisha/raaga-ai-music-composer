"""Unit tests: settings, credentials, project storage and crash recovery."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from raagacomposer.core.models import (ApprovalState, ConversationTurn,
                                       ErrorRecord, JobRecord, MelodyVersion,
                                       Note, Project, Section, Stage)
from raagacomposer.core.persistence import (BACKUP_FILE, PROJECT_FILE,
                                            ProjectStore, slugify)
from raagacomposer.core.settings import Settings, config_dir

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
def test_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    settings = Settings.load()
    settings.sample_rate = 48000
    settings.theme = "dark"
    settings.save()
    reloaded = Settings.load()
    assert reloaded.sample_rate == 48000
    assert Path(reloaded.projects_dir).name == "Projects"


def test_settings_survive_a_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    (tmp_path / "settings.json").write_text("{ broken", encoding="utf-8")
    settings = Settings.load()
    assert settings.sample_rate == 44100          # defaults, not a crash


def test_settings_ignore_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    (tmp_path / "settings.json").write_text(
        json.dumps({"sample_rate": 22050, "from_the_future": True}),
        encoding="utf-8")
    assert Settings.load().sample_rate == 22050


def test_secrets_prefer_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    Settings.set_secret("anthropic_api_key", "from-file")
    assert Settings.secret("anthropic_api_key") == "from-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert Settings.secret("anthropic_api_key") == "from-env"


def test_secrets_are_absent_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Settings.secret("anthropic_api_key") == ""


def test_no_key_is_ever_hard_coded():
    source = Path(__file__).resolve().parents[2] / "raagacomposer"
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sk-ant-" not in text, path
        assert "api_key=\"" not in text.replace('api_key=""', ""), path


def test_recent_projects_are_remembered_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    settings = Settings.load()
    settings.recent_projects = []
    settings.remember_project("a")
    settings.remember_project("b")
    settings.remember_project("a")
    assert settings.recent_projects[:2] == ["a", "b"]
    settings.forget_project("a")
    assert "a" not in settings.recent_projects


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def test_slugify_makes_a_safe_folder_name():
    assert slugify("Terrace at Midnight!") == "terrace-at-midnight"
    assert slugify("") == "song"


def test_create_lays_out_the_project_folder(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Night Song")
    assert (directory / PROJECT_FILE).exists()
    for sub in ("audio", "renders", "mixes", "exports", "voices"):
        assert (directory / sub).is_dir()
    assert project.history and project.history[0].action == "project.create"


def test_save_and_open_round_trip(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Round Trip")
    section = Section(name="Pallavi", start=0.0, end=8.0, locked=True)
    project.melodies = [MelodyVersion(version=1, raaga="Mohanam",
                                      sections=[section],
                                      notes=[Note(swara="G3", midi=64,
                                                  section_id=section.id)],
                                      state=ApprovalState.LOCKED)]
    project.approved_melody = 1
    project.raaga.selected = "Mohanam"
    project.current_stage = Stage.ARRANGEMENT
    store.save(project, directory)

    reopened = store.open(directory)
    assert reopened.title == "Round Trip"
    assert reopened.raaga.selected == "Mohanam"
    assert reopened.current_stage is Stage.ARRANGEMENT
    assert reopened.melody().state is ApprovalState.LOCKED
    assert reopened.melody().sections[0].locked
    assert reopened.melody().notes[0].swara == "G3"


def test_saving_writes_a_backup_copy(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Backups")
    store.save(project, directory)
    project.title = "Backups v2"
    store.save(project, directory)
    assert (directory / BACKUP_FILE).exists()
    backup = json.loads((directory / BACKUP_FILE).read_text(encoding="utf-8"))
    assert backup["title"] == "Backups"


def test_a_corrupt_project_file_is_recovered_from_the_backup(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Crash Test")
    store.save(project, directory)
    project.title = "Crash Test v2"
    store.save(project, directory)

    (directory / PROJECT_FILE).write_text("{ not json at all", encoding="utf-8")
    recovered = store.open(directory)
    assert recovered.title == "Crash Test"

    with sqlite3.connect(str(directory / "project.db")) as conn:
        rows = conn.execute("SELECT message FROM errors").fetchall()
    assert any("unreadable" in row[0] for row in rows)


def test_opening_a_folder_with_nothing_readable_raises(settings, tmp_path):
    store = ProjectStore(settings)
    with pytest.raises(FileNotFoundError):
        store.open(tmp_path / "empty")


def test_opening_accepts_the_project_file_itself(settings):
    store = ProjectStore(settings)
    project, directory = store.create("By File")
    assert store.open(directory / PROJECT_FILE).title == "By File"


def test_the_journal_records_history_and_conversation(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Journal")
    project.log_history("tune.generate", "Generated tune v1")
    project.conversation.append(ConversationTurn(text="add veena here",
                                                 intent="arrange.add",
                                                 status="applied"))
    project.jobs.append(JobRecord(job_type="render.full", target="render:full",
                                  status="done"))
    project.errors.append(ErrorRecord(where="render", message="device missing"))
    store.save(project, directory)

    with sqlite3.connect(str(directory / "project.db")) as conn:
        assert conn.execute("SELECT count(*) FROM history").fetchone()[0] >= 2
        assert conn.execute("SELECT text FROM conversation").fetchone()[0] == \
            "add veena here"
        assert conn.execute("SELECT job_type FROM jobs").fetchone()[0] == \
            "render.full"
        assert conn.execute("SELECT message FROM errors").fetchone()[0] == \
            "device missing"
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta["title"] == "Journal"


def test_artifacts_are_recorded(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Artifacts")
    path = ProjectStore.artifact_path(directory, "mixes", "full_v1.wav")
    path.write_bytes(b"RIFF")
    store.note_artifact(directory, path, "mix", "first mix")
    with sqlite3.connect(str(directory / "project.db")) as conn:
        rows = conn.execute("SELECT path, kind FROM artifacts").fetchall()
    assert rows and rows[0][1] == "mix"


def test_the_registry_lists_recent_projects(settings):
    store = ProjectStore(settings)
    first, dir_a = store.create("Alpha")
    second, dir_b = store.create("Beta")
    titles = [entry["title"] for entry in store.recent()]
    assert "Alpha" in titles and "Beta" in titles


def test_save_as_copies_the_artifacts(settings, tmp_path):
    store = ProjectStore(settings)
    project, directory = store.create("Original")
    ProjectStore.artifact_path(directory, "mixes", "full_v1.wav").write_bytes(b"x")
    target = tmp_path / "copy"
    store.save_as(project, directory, target)
    assert (target / PROJECT_FILE).exists()
    assert (target / "mixes" / "full_v1.wav").exists()
    assert any(h.action == "project.save_as" for h in project.history)


def test_an_atomic_save_leaves_no_temporary_file(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Atomic")
    store.save(project, directory)
    assert not list(directory.glob("*.tmp"))


# --------------------------------------------------------------------------
# provenance (docs/PLAN_agent_factory.md, "Item 4, integrated")
# --------------------------------------------------------------------------
def test_melody_provenance_survives_save_and_reopen(settings):
    store = ProjectStore(settings)
    project, directory = store.create("Provenance Round Trip")
    section = Section(name="Pallavi", start=0.0, end=8.0)
    melody = MelodyVersion(
        version=1, raaga="Keeravani", sections=[section],
        notes=[Note(swara="S", midi=60, section_id=section.id),
              Note(swara="R2", midi=62, section_id=section.id),
              Note(swara="G2", midi=64, section_id=section.id)],
        provenance=[{"start": 0, "end": 2, "swaras": "S R2 G2",
                    "source": "learned", "phrase_id": "phr_abc",
                    "origin": "a lesson recording", "section_id": section.id}],
        guidance_note="avoid large leaps")
    project.melodies = [melody]
    project.approved_melody = 1
    store.save(project, directory)

    reopened = store.open(directory)
    got = reopened.melody()
    assert got.provenance == melody.provenance
    assert got.guidance_note == "avoid large leaps"


def test_a_project_saved_without_provenance_loads_with_an_empty_list(settings):
    """A project.json written before this field existed has no ``provenance``
    key at all; ``core.serde.from_jsonable`` must tolerate that rather than
    fail to load an old project."""
    store = ProjectStore(settings)
    project, directory = store.create("Old Project")
    section = Section(name="Pallavi", start=0.0, end=8.0)
    project.melodies = [MelodyVersion(version=1, raaga="Mohanam",
                                      sections=[section],
                                      notes=[Note(swara="G3", midi=64,
                                                  section_id=section.id)])]
    project.approved_melody = 1
    store.save(project, directory)

    path = directory / PROJECT_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    melody_data = data["melodies"][0]
    assert "provenance" in melody_data      # sanity: the field is written
    del melody_data["provenance"]
    del melody_data["guidance_note"]
    path.write_text(json.dumps(data), encoding="utf-8")

    reopened = store.open(directory)
    assert reopened.melody().provenance == []
    assert reopened.melody().guidance_note == ""
