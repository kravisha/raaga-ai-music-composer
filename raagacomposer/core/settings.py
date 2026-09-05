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
    "youtube_api_key": "YOUTUBE_API_KEY",
}


@dataclass
class Settings:
    projects_dir: str = ""
    # auto | off | claude | ollama | llamacpp.  "local" and "anthropic" are
    # accepted as the older names for "off" and "claude".
    llm_provider: str = "auto"
    # How the router chooses between the backends that are up:
    # auto | local_first | claude_first | local_only | claude_only | off.
    #
    # local_first is the standing policy and the default: attempt a local
    # model for every task, judge what it produced, and reach a paid model
    # only when the local answer has actually been found wanting.  It is the
    # default rather than merely available because the point of the policy is
    # that local is tried by default, not on request - and it is safe to
    # default to only because the judge exists to catch a bad local answer
    # (providers/escalation.py).  "auto" keeps the older behaviour, where
    # complexity and cost order the backends and a strength floor stands in
    # for a judge.
    llm_routing: str = "local_first"
    llm_claude_model: str = "claude-opus-5"        # quality-critical work
    llm_claude_light_model: str = "claude-haiku-4-5"   # cheap, high-volume work
    llm_claude_effort: str = "medium"              # low | medium | high | xhigh | max
    llm_claude_thinking: bool = True               # adaptive, on hard tasks only
    llm_local_endpoint: str = "http://127.0.0.1:11434"
    llm_local_model: str = "qwen3:4b"
    llm_local_gguf: str = ""            # blank = first *.gguf in <config>/models/llm
    llm_local_strength: int = 40        # rough capability of the local model, 0-100

    # --- local-first routing ------------------------------------------
    # The whole policy in one block, so changing it needs no code edit and
    # no redeploy.  ``llm_routing`` above is the mode:
    #
    #   local_first   attempt local, escalate to a paid model only on a
    #                 judged failure (the standing default)
    #   local_only    attempt local, never escalate
    #   claude_only   skip the local attempt entirely rather than running
    #                 it and discarding the result, so a rollback costs
    #                 nothing in latency
    #
    # Every mode produces the same output shape, so nothing downstream
    # cares which model ran - but the mode and the model are recorded with
    # each result, because otherwise a quality dip cannot be told apart
    # from a change we made ourselves.
    #
    # Tiers are named rather than ordered so the escalation loop can ask
    # for the one it wants.  "json" is a specialist, not a rung: a task
    # whose answer must validate against a schema - the fourteen-dimension
    # affect vector of raaga/emotion.py - starts there instead of at the
    # cheapest.
    routing_tiers: Dict[str, str] = field(default_factory=lambda: {
        "small": "qwen3:4b",       # 2.3 GB, the cheap first attempt
        "mid": "qwen3:8b",         # 4.9 GB, the step up before anything paid
        "json": "hermes3:8b",      # 4.3 GB, tuned for structured output
    })
    #: Escalation order for ordinary prose work, cheapest first.
    routing_order: list = field(default_factory=lambda: ["small", "mid"])
    #: Escalation order when the answer must validate against a schema.
    #: A different family from the other two on purpose, so that the
    #: two-sample divergence check below is a second opinion rather than
    #: the same model agreeing with itself.
    routing_order_json: list = field(default_factory=lambda: ["json", "mid"])

    # The judge, in the order the policy applies it.  Schema validity is
    # pass/fail and needs no threshold; these are the other two, plus a
    # deadline, because a judge that tests only quality lets a local model
    # spend ten minutes before escalating.
    routing_logprob_floor: float = -1.10   # mean token logprob; below = low confidence
    routing_divergence: float = 0.15       # two samples differing by more than this fail
    routing_sample_temperature: float = 0.5
    routing_attempt_seconds: float = 90.0  # exceeded = a failure, escalate
    #: Every attempt is written here with the brief, the local output, the
    #: failing signal and the paid output, so the thresholds above can be
    #: tuned against real cases instead of guesses.
    routing_log: str = ""                  # blank = <config>/routing_attempts.jsonl
    # A small model on a CPU is slow: writing a full lyric takes far longer
    # than any remote call. Too low a ceiling here does not fail the request,
    # it silently routes the work away from the local model.
    llm_local_timeout: float = 180.0
    llm_refresh_seconds: float = 30.0   # how often to notice a key being added
    stt_provider: str = "auto"          # auto | none | vosk | whisper
    # Which Whisper to use.  Measured on this machine, transcribing a
    # three-second phrase on the CPU: tiny 0.8s, base 10.6s.  A voice
    # command you wait ten seconds for is not a voice command, and a
    # misheard one you can simply repeat, so the small fast model is the
    # default.  Raise it to "base" or "small" if accuracy matters more than
    # waiting - dictating lyrics, say, rather than saying "add a violin".
    stt_model_size: str = "tiny"        # tiny | base | small | medium
    #: What the tune and the audition are heard on when the brief does
    #: not name a lead instrument.  Any catalog key with a "lead"
    #: role: veena, violin, flute, sitar, sarod, santoor, piano...
    tune_instrument: str = "veena"
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
    factory_db: str = ""                   # blank = <config>/factory.db
    learning_corpus_dir: str = ""          # your own audio to learn from
    learning_allow_web: bool = False       # record leads only; never fetches
    learning_autostart: bool = False
    learning_max_sources_per_lesson: int = 4
    # A whole recording, not the first two minutes of one.  At 120s a
    # 15.7-minute concert or film-song mashup was 87% unheard, and what
    # survived was whatever happened to be at the start - an introduction,
    # an announcement, a title card.  The cost of listening to the rest is
    # analysis time, about a minute of CPU per recording, not storage.
    # Set it to 0 to place no limit at all.
    learning_max_audio_seconds: float = 1800.0
    learning_max_steps_per_session: int = 200
    learning_step_pause: float = 0.5
    learning_min_confidence: float = 0.35
    learning_max_storage_mb: int = 512
    # Real recordings carry a drone and a teacher talking; rendered exercises
    # carry neither, so preparation is applied to supplied audio only.
    # --- training tab ---------------------------------------------------
    training_db: str = ""                  # blank = <config>/training.db
    # The permanent learned memory.  Blank = <config>/knowledge_base.db.
    # It is opened, never recreated: see docs and kb/store.py.
    knowledge_base_db: str = ""
    training_allow_web: bool = False       # leads only; never fetches
    training_max_results: int = 10
    learning_preprocess_recordings: bool = True
    learning_remove_drone: bool = True
    learning_gate_speech: bool = True
    # --- guided composition (docs/PLAN_learning_loop.md item 4) ---------
    compose_rewrites: int = 3          # rewrites tried before keeping the best
    compose_threshold: float = 0.7     # evaluator overall score to pass
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
    # Resolution order, and the keyring-then-file storage below the
    # environment, live in core.secrets.SecretStore; these two classmethods
    # are kept as the stable entry point every caller already uses.
    @staticmethod
    def credentials_path() -> Path:
        from .secrets import credentials_path as _path
        return _path()

    @classmethod
    def secret(cls, name: str) -> str:
        from .secrets import SecretStore
        return SecretStore().get(name)

    @classmethod
    def set_secret(cls, name: str, value: str) -> None:
        from .secrets import SecretStore
        SecretStore().set(name, value)

    # -- recents -----------------------------------------------------------
    def remember_project(self, path: str) -> None:
        path = str(path)
        self.recent_projects = [path] + [p for p in self.recent_projects if p != path]
        del self.recent_projects[12:]
        self.save()

    def forget_project(self, path: str) -> None:
        self.recent_projects = [p for p in self.recent_projects if p != str(path)]
        self.save()
