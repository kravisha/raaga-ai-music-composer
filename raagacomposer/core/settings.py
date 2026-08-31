"""Settings and credential management (spec section 12.33, 11).

API keys are never hard-coded.  Resolution order for any secret:

1. environment variable
2. ``credentials.json`` in the application config directory
3. absent -> the provider reports itself unavailable and the app falls back to
   the built-in local engines.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

APP_NAME = "RaagaComposer"


def config_dir() -> Path:
    base = os.environ.get("RAAGA_COMPOSER_HOME")
    if base:
        p = Path(base)
    elif os.name == "nt":
        p = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    else:
        p = Path.home() / f".{APP_NAME.lower()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_projects_dir() -> Path:
    docs = Path.home() / "Documents"
    base = (docs if docs.exists() else Path.home()) / APP_NAME / "Projects"
    base.mkdir(parents=True, exist_ok=True)
    return base


ENV_KEYS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "elevenlabs_api_key": "ELEVENLABS_API_KEY",
    "deepgram_api_key": "DEEPGRAM_API_KEY",
}


@dataclass
class Settings:
    projects_dir: str = ""
    llm_provider: str = "auto"          # auto | local | anthropic
    stt_provider: str = "auto"          # auto | none | vosk | whisper
    music_provider: str = "local"
    voice_provider: str = "local"
    sample_rate: int = 44100
    autosave_seconds: int = 30
    mic_device: str = ""
    output_device: str = ""
    vad_threshold: float = 0.012
    vad_silence_ms: int = 700
    listen_on_start: bool = False
    theme: str = "dark"
    log_level: str = "INFO"
    # --- learning agent -------------------------------------------------
    pilot_raaga: str = "Keeravani"
    knowledge_db: str = ""                 # blank = <config>/knowledge.db
    learning_corpus_dir: str = ""          # your own audio to learn from
    learning_allow_web: bool = False       # record leads only; never fetches
    learning_autostart: bool = False
    learning_max_sources_per_lesson: int = 4
    learning_max_audio_seconds: float = 120.0
    learning_max_steps_per_session: int = 200
    learning_step_pause: float = 0.5
    learning_min_confidence: float = 0.35
    learning_max_storage_mb: int = 512
    recent_projects: list = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def path(cls) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        p = cls.path()
        data: Dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        known = {f for f in cls.__dataclass_fields__}
        s = cls(**{k: v for k, v in data.items() if k in known})
        if not s.projects_dir:
            s.projects_dir = str(default_projects_dir())
        return s

    def save(self) -> None:
        p = self.path()
        tmp = p.with_suffix(".tmp")
        payload = {f: getattr(self, f) for f in self.__dataclass_fields__}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)

    # -- credentials -------------------------------------------------------
    @staticmethod
    def credentials_path() -> Path:
        return config_dir() / "credentials.json"

    @classmethod
    def secret(cls, name: str) -> str:
        env = ENV_KEYS.get(name, name.upper())
        val = os.environ.get(env, "")
        if val:
            return val.strip()
        p = cls.credentials_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return str(data.get(name, "")).strip()
            except Exception:
                return ""
        return ""

    @classmethod
    def set_secret(cls, name: str, value: str) -> None:
        p = cls.credentials_path()
        data: Dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if value:
            data[name] = value
        else:
            data.pop(name, None)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass

    # -- recents -----------------------------------------------------------
    def remember_project(self, path: str) -> None:
        path = str(path)
        self.recent_projects = [path] + [p for p in self.recent_projects if p != path]
        del self.recent_projects[12:]
        self.save()

    def forget_project(self, path: str) -> None:
        self.recent_projects = [p for p in self.recent_projects if p != str(path)]
        self.save()
