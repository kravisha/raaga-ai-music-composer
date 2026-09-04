"""The provider manager: which backend answers which task, and what happens
when none of them can.

Nothing here touches the network or a real model.  The backends are stand-ins
that record what they were asked, because what is being tested is the routing
decision itself.
"""
from __future__ import annotations

import json

import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.core.settings import Settings
from raagacomposer.providers import prompts, registry, tasks
from raagacomposer.providers.base import LLMProvider
from raagacomposer.providers.claude_llm import MODELS, ClaudeLLM
from raagacomposer.providers.local import LocalLLM
from raagacomposer.providers.local_llm import LlamaCppLLM, OllamaLLM
from raagacomposer.providers.router import RoutedLLM

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------
class FakeBackend(LLMProvider):
    def __init__(self, name, strength, cost, local, ok=True, fails=False,
                 empty=False):
        self.name = name
        self.strength = strength
        self.cost_per_mtok = cost
        self.is_local = local
        self._ok = ok
        self.fails = fails
        self.empty = empty
        self.calls = []

    @property
    def available(self):
        return self._ok

    def _run(self, what, value):
        self.calls.append(what)
        if self.fails:
            raise RuntimeError("backend down")
        return type(value)() if self.empty else value

    def write_lyrics(self, slots, brief):
        return self._run("write_lyrics", [f"line from {self.name}"])

    def classify_intent(self, text, intents):
        return self._run("classify_intent", {"intent": "play", "by": self.name})

    def suggest_raagas(self, brief, candidates):
        return self._run("suggest_raagas", [{"raaga": "K", "reason": self.name}])

    def suggest_instruments(self, description, catalog):
        return self._run("suggest_instruments", ["veena"])

    def explain(self, question, context=""):
        return self._run("explain", f"answer from {self.name}")


def trio():
    """A strong cloud model, a cheap cloud model and a small local one."""
    return (FakeBackend("opus", 95, 10.0, local=False),
            FakeBackend("haiku", 65, 2.0, local=False),
            FakeBackend("local3b", 40, 0.0, local=True))


def router(*backends, policy="auto"):
    return RoutedLLM([lambda b=b: b for b in backends], policy=policy,
                     refresh_seconds=0)


# --------------------------------------------------------------------------
# the task taxonomy
# --------------------------------------------------------------------------
def test_every_provider_method_has_a_task_spec():
    methods = {m for m in dir(LLMProvider)
               if not m.startswith("_") and callable(getattr(LLMProvider, m))
               and m not in ("info", "status")}
    assert methods <= set(tasks.TASKS)


def test_lyrics_is_the_hard_task_and_intent_is_the_urgent_one():
    assert tasks.spec(tasks.WRITE_LYRICS).complexity is tasks.Complexity.HIGH
    assert tasks.spec(tasks.WRITE_LYRICS).quality_critical
    assert tasks.spec(tasks.CLASSIFY_INTENT).latency_critical
    assert not tasks.spec(tasks.EXPLAIN).wants_json


# --------------------------------------------------------------------------
# routing on complexity, cost and locality
# --------------------------------------------------------------------------
def test_a_hard_task_goes_to_the_strongest_backend():
    opus, haiku, local = trio()
    r = router(local, haiku, opus)                # deliberately worst-first
    assert r.write_lyrics([], CreativeBrief()) == ["line from opus"]
    assert r.suggest_raagas(CreativeBrief(), [])[0]["reason"] == "opus"


def test_an_easy_task_goes_to_the_cheapest_backend():
    opus, haiku, local = trio()
    r = router(opus, haiku, local)
    r.suggest_instruments("warm", ["veena"])
    assert r.last_route[tasks.SUGGEST_INSTRUMENTS] == "local3b"


def test_an_urgent_task_prefers_a_local_backend_over_a_stronger_remote_one():
    opus, haiku, local = trio()
    r = router(opus, haiku, local)
    assert r.classify_intent("add veena", [])["by"] == "local3b"
    assert opus.calls == [] and haiku.calls == []


def test_a_middling_task_prefers_the_cheap_capable_model_not_the_weak_one():
    opus, haiku, local = trio()
    r = router(opus, haiku, local)
    assert r.explain("why this phrase?") == "answer from haiku"


def test_a_weak_local_model_is_still_the_fallback_for_a_middling_task():
    _, _, local = trio()
    assert router(local).explain("q") == "answer from local3b"


def test_a_weak_backend_is_not_asked_to_write_lyrics_in_an_unjudged_mode(caplog):
    """Measured: a 3B model spent 704s on ten lines and returned nothing
    usable, while the built-in engine fits them exactly and instantly.

    Where there is no judge to catch that, the strength floor still keeps it
    out of the quality-critical work; the judged modes attempt it and decide
    on the answer instead (see the local_first tests below).
    """
    _, _, local = trio()
    r = router(local, policy="auto")
    assert r.chain(tasks.WRITE_LYRICS) == []
    assert r.chain(tasks.SUGGEST_RAAGAS) == []
    assert r.write_lyrics([], CreativeBrief()) == []
    assert local.calls == [], "the weak backend must not be tried"
    # ...but it is still perfectly good for the easy work.
    assert r.classify_intent("x", [])["by"] == "local3b"


def test_a_stronger_local_model_does_qualify_for_lyrics():
    """The floor is on capability, not on being local."""
    big = FakeBackend("local70b", 80, 0.0, local=True)
    r = router(big)
    assert r.write_lyrics([], CreativeBrief()) == ["line from local70b"]


def test_the_quality_floor_is_adjustable():
    _, _, local = trio()
    r = RoutedLLM([lambda: local], refresh_seconds=0, quality_floor=10)
    assert r.write_lyrics([], CreativeBrief()) == ["line from local3b"]


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("policy,first", [
    ("auto", "local3b"),          # an easy task goes to the cheapest
    ("local_first", "local3b"),
    ("claude_first", "haiku"),    # cheapest of the remote pair
    ("local_only", "local3b"),
    ("claude_only", "haiku"),
])
def test_the_policy_reorders_the_chain(policy, first):
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy=policy)
    assert r.chain(tasks.SUGGEST_INSTRUMENTS)[0].name == first


@pytest.mark.parametrize("policy,first", [
    ("auto", "opus"),
    ("claude_first", "opus"),
    ("claude_only", "opus"),
])
def test_a_hard_task_ignores_a_preference_for_a_backend_too_weak_for_it(policy,
                                                                       first):
    """In the unjudged modes the strength floor is all there is to go on, so
    a weak backend is still kept out of the quality-critical work."""
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy=policy)
    assert r.chain(tasks.WRITE_LYRICS)[0].name == first


def test_a_judged_mode_tries_the_weak_local_model_before_anything_paid():
    """The standing policy replaced the floor here: nothing is excluded
    before it has been tried, and the judge decides on what came back
    rather than on a strength number chosen in advance."""
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy="local_first")
    chain = r.chain(tasks.WRITE_LYRICS)
    assert chain[0].name == "local3b"
    assert [b.name for b in chain[1:]] == ["opus", "haiku"]


def test_local_only_now_attempts_the_weak_model_rather_than_excluding_it():
    """It may still end at the built-in engine - but by being judged and
    found wanting, not by never being asked."""
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy="local_only")
    assert [b.name for b in r.chain(tasks.WRITE_LYRICS)] == ["local3b"]
    assert r.write_lyrics([], CreativeBrief()) == ["line from local3b"]
    assert local.calls, "the local model must actually be asked"


def test_local_only_never_reaches_a_remote_backend():
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy="local_only")
    r.write_lyrics([], CreativeBrief())
    assert opus.calls == [] and haiku.calls == []


def test_claude_only_never_reaches_a_local_backend():
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy="claude_only")
    r.classify_intent("x", [])                    # would prefer local on auto
    assert local.calls == []


def test_off_uses_the_built_in_engines_even_with_backends_up():
    opus, haiku, local = trio()
    r = router(opus, haiku, local, policy="off")
    assert not r.available
    assert r.write_lyrics([], CreativeBrief()) == []
    assert opus.calls == []


def test_an_unknown_policy_falls_back_to_auto():
    assert router(*trio(), policy="sideways").policy == "auto"


def test_a_backend_ruled_out_by_policy_says_so_rather_than_looking_absent():
    """Having none and being forbidden all of them are different faults."""
    opus, haiku, local = trio()
    r = router(opus, haiku, policy="local_only")   # both remote, none local
    assert not r.available
    assert "excludes" in r.status() and "local_only" in r.status()


# --------------------------------------------------------------------------
# failure and the floor beneath everything
# --------------------------------------------------------------------------
def test_a_failing_backend_hands_the_task_to_the_next_one():
    opus = FakeBackend("opus", 95, 10.0, local=False, fails=True)
    haiku = FakeBackend("haiku", 65, 2.0, local=False)
    r = router(opus, haiku)
    assert r.write_lyrics([], CreativeBrief()) == ["line from haiku"]
    assert opus.calls == ["write_lyrics"]         # it was tried, and it failed


def test_a_backend_that_answers_with_nothing_is_not_the_answer():
    opus = FakeBackend("opus", 95, 10.0, local=False, empty=True)
    haiku = FakeBackend("haiku", 65, 2.0, local=False)
    r = router(opus, haiku)
    assert r.write_lyrics([], CreativeBrief()) == ["line from haiku"]


def test_an_unavailable_backend_is_never_called():
    dead = FakeBackend("dead", 95, 1.0, local=False, ok=False)
    live = FakeBackend("live", 40, 0.0, local=True)
    r = router(dead, live)
    r.write_lyrics([], CreativeBrief())
    assert dead.calls == []


def test_with_every_backend_gone_the_router_reports_unavailable_and_returns_empty():
    dead = FakeBackend("dead", 95, 1.0, local=False, ok=False)
    r = router(dead)
    assert not r.available
    assert r.write_lyrics([], CreativeBrief()) == []
    assert r.classify_intent("x", []) == {}
    assert r.suggest_raagas(CreativeBrief(), []) == []
    assert r.suggest_instruments("warm", []) == []
    assert r.explain("q") == ""
    assert "rule and lexicon engines" in r.status()


def test_a_backend_that_fails_to_construct_does_not_take_the_router_down():
    def explode():
        raise RuntimeError("no runtime here")

    good = FakeBackend("good", 50, 1.0, local=True)
    r = RoutedLLM([explode, lambda: good], refresh_seconds=0)
    assert r.available
    assert r.explain("q") == "answer from good"


def test_routing_can_be_explained():
    text = router(*trio()).explain_routing()
    assert "write_lyrics" in text and "opus" in text
    assert "classify_intent" in text and "local3b" in text


# --------------------------------------------------------------------------
# enabling Claude later, without a code change
# --------------------------------------------------------------------------
def test_a_key_added_while_running_switches_claude_on(monkeypatch):
    """The requirement in one test: no key, then a key, no restart."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def factory():
        return FakeBackend("claude", 95, 10.0, local=False,
                           ok=bool(Settings.secret("anthropic_api_key")))

    r = RoutedLLM([factory], refresh_seconds=30.0)
    assert not r.available
    assert r.write_lyrics([], CreativeBrief()) == []

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    r._checked = 0.0                              # let the throttle expire
    assert r.available
    assert r.write_lyrics([], CreativeBrief()) == ["line from claude"]


def test_an_explicit_refresh_also_picks_up_a_new_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = RoutedLLM([lambda: FakeBackend(
        "claude", 95, 10.0, local=False,
        ok=bool(Settings.secret("anthropic_api_key")))], refresh_seconds=0)
    assert not r.available
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    r.refresh()
    assert r.available


# --------------------------------------------------------------------------
# the Claude adapter
# --------------------------------------------------------------------------
class FakeBlock:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [FakeBlock("thinking"), FakeBlock("text", text)]
        self.stop_reason = stop_reason
        self.stop_details = None


class FakeMessages:
    def __init__(self, text):
        self.text = text
        self.seen = {}

    def create(self, **kwargs):
        self.seen = kwargs
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text):
        self.messages = FakeMessages(text)


def claude(model, text, **kw):
    llm = ClaudeLLM(model=model, api_key="sk-ant-not-a-real-key", **kw)
    llm._client = FakeClient(text)
    return llm


def test_the_model_table_prices_and_ranks_the_tiers_consistently():
    opus, sonnet, haiku = (MODELS["claude-opus-5"], MODELS["claude-sonnet-5"],
                           MODELS["claude-haiku-4-5"])
    assert opus.strength > sonnet.strength > haiku.strength
    assert opus.cost > sonnet.cost > haiku.cost


def test_effort_and_thinking_are_only_sent_to_models_that_accept_them():
    """Sending either to an older model is a 400, so this is correctness."""
    assert MODELS["claude-opus-5"].effort
    assert not MODELS["claude-haiku-4-5"].effort

    modern = claude("claude-opus-5", '["a line"]')
    modern.write_lyrics([], CreativeBrief())
    assert modern._client.messages.seen["output_config"] == {"effort": "medium"}
    assert modern._client.messages.seen["thinking"] == {"type": "adaptive"}

    older = claude("claude-haiku-4-5", '["a line"]')
    older.write_lyrics([], CreativeBrief())
    assert "output_config" not in older._client.messages.seen
    assert "thinking" not in older._client.messages.seen


def test_thinking_is_kept_off_the_easy_tasks_and_given_room_on_the_hard_ones():
    llm = claude("claude-opus-5", '{"intent": "play"}')
    llm.classify_intent("play it", ["play"])
    easy = llm._client.messages.seen
    assert "thinking" not in easy
    assert easy["max_tokens"] == tasks.TASKS[tasks.CLASSIFY_INTENT].max_tokens

    llm = claude("claude-opus-5", '["a line"]')
    llm.write_lyrics([], CreativeBrief())
    hard = llm._client.messages.seen
    assert hard["thinking"] == {"type": "adaptive"}
    assert hard["max_tokens"] > tasks.TASKS[tasks.WRITE_LYRICS].max_tokens


def test_thinking_blocks_are_not_mistaken_for_the_answer():
    llm = claude("claude-opus-5", '["only this"]')
    assert llm.write_lyrics([], CreativeBrief()) == ["only this"]


def test_a_refusal_is_raised_so_the_router_can_try_the_next_backend():
    llm = claude("claude-opus-5", "no")
    llm._client.messages.create = lambda **kw: FakeResponse("", "refusal")
    with pytest.raises(RuntimeError):
        llm.explain("something")


def test_without_a_key_the_claude_adapter_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, name: ""))
    llm = ClaudeLLM()
    assert not llm.available
    assert "No API key" in llm.status()
    with pytest.raises(RuntimeError):
        llm.explain("q")


# --------------------------------------------------------------------------
# the local adapters
# --------------------------------------------------------------------------
def test_ollama_is_unavailable_when_no_server_is_listening():
    llm = OllamaLLM(endpoint="http://127.0.0.1:1")
    assert not llm.available
    assert "no Ollama server" in llm.status()
    with pytest.raises(RuntimeError):
        llm.explain("q")


class FakeHTTP:
    """Stands in for urlopen so the probe can be driven without a server."""

    def __init__(self, payload):
        self.payload = payload

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        import json
        return json.dumps(self.payload).encode("utf-8")


def serving(monkeypatch, *model_names):
    monkeypatch.setattr(
        "raagacomposer.providers.local_llm.urllib.request.urlopen",
        FakeHTTP({"models": [{"name": n} for n in model_names]}))


def test_ollama_is_ready_when_the_server_has_the_model(monkeypatch):
    serving(monkeypatch, "llama3.2:3b", "mistral:7b")
    llm = OllamaLLM(model="llama3.2:3b")
    assert llm.available
    assert "llama3.2:3b" in llm.status()


def test_a_tagless_name_matches_the_tagged_model(monkeypatch):
    serving(monkeypatch, "llama3.2:latest")
    assert OllamaLLM(model="llama3.2").available


def test_ollama_names_the_command_that_would_fix_a_missing_model(monkeypatch):
    """A model the creator did not ask for is never quietly substituted."""
    serving(monkeypatch, "mistral:7b")               # up, but not what we want
    llm = OllamaLLM(model="llama3.2:3b")
    assert not llm.available, "must not answer with a different model"
    assert "ollama pull llama3.2:3b" in llm.status()


def test_a_running_server_is_reported_differently_from_no_server(monkeypatch):
    serving(monkeypatch, "mistral:7b")
    running = OllamaLLM(model="llama3.2:3b").status()
    monkeypatch.undo()                    # the stub answers any endpoint
    absent = OllamaLLM(endpoint="http://127.0.0.1:1").status()
    assert "is running but" in running
    assert "no Ollama server" in absent


def test_llamacpp_is_unavailable_without_the_package_or_a_model():
    llm = LlamaCppLLM()
    assert not llm.available
    assert llm.status()                          # says which of the two is missing


def test_the_local_backends_are_free_and_declare_themselves_local():
    for llm in (OllamaLLM(endpoint="http://127.0.0.1:1"), LlamaCppLLM()):
        assert llm.is_local and llm.cost_per_mtok == 0.0


# --------------------------------------------------------------------------
# parsing what a model sends back
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ('["a", "b"]', ["a", "b"]),
    ('```json\n["a", "b"]\n```', ["a", "b"]),
    ('Sure! Here you go:\n["a", "b"]\nHope that helps.', ["a", "b"]),
    ('{"lines": ["a", "b"]}', ["a", "b"]),
    # Constrained to emit an object, a small model numbers the keys instead
    # of returning an array.  Observed from llama3.2:3b, not imagined.
    ('{"1": "a\\n", "2": "b\\n"}', ["a", "b"]),
    ('{"10": "j", "2": "b", "1": "a"}', ["a", "b", "j"]),   # numeric, not lexical
    ("not json at all", []),
    ("", []),
])
def test_lyrics_are_read_back_from_whatever_shape_the_model_used(text, expected):
    assert prompts.as_lyrics(prompts.extract_json(text)) == expected


def test_a_single_raaga_suggestion_is_still_a_suggestion():
    """llama3.2:3b answers with one object where a list was asked for."""
    data = prompts.extract_json('{"raaga": "Bhairavi", "reason": "longing"}')
    assert prompts.as_raagas(data) == [{"raaga": "Bhairavi", "reason": "longing"}]


def test_raagas_are_read_back_from_the_wrapped_shape():
    data = prompts.extract_json('{"raagas": [{"raaga": "K", "reason": "r"}]}')
    assert prompts.as_raagas(data) == [{"raaga": "K", "reason": "r"}]


def test_instruments_are_read_back_from_the_wrapped_and_numbered_shapes():
    wrapped = prompts.extract_json('{"instruments": ["veena", "tuba"]}')
    assert prompts.as_instruments(wrapped, ["veena", "flute"]) == ["veena"]
    numbered = prompts.extract_json('{"1": "flute", "2": "veena"}')
    assert prompts.as_instruments(numbered, ["veena", "flute"]) == ["flute", "veena"]


def test_the_prompts_name_the_json_shape_they_expect():
    """Ollama's JSON mode emits an object, so asking for a bare array fails."""
    for system, _ in (prompts.lyrics([], CreativeBrief()),
                      prompts.raagas(CreativeBrief(), ["K"]),
                      prompts.instruments("warm", ["veena"])):
        assert "{" in system and "}" in system, system


def test_an_instrument_outside_the_catalog_is_dropped():
    data = prompts.extract_json('["veena", "theremin", {"instrument": "flute"}]')
    assert prompts.as_instruments(data, ["veena", "flute"]) == ["veena", "flute"]


def test_every_backend_asks_the_same_question():
    """One task is framed one way, whoever answers it."""
    system, user = prompts.lyrics([], CreativeBrief(language="Tamil"))
    assert "syllables" in system and "Tamil" in user


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------
def test_switching_the_llm_off_returns_the_built_in_engines(settings):
    settings.llm_provider = "off"
    assert isinstance(registry.build_llm(settings), LocalLLM)


def test_the_older_setting_names_still_work(settings):
    settings.llm_provider = "local"               # used to mean "no model"
    assert isinstance(registry.build_llm(settings), LocalLLM)


def test_claude_only_builds_no_local_backends(settings, monkeypatch):
    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, n: ""))
    settings.llm_provider = "claude"
    llm = registry.build_llm(settings)
    assert all(not b.is_local for b in llm.backends)
    assert not llm.available                      # no key, so nothing is up


def test_the_registry_builds_both_claude_tiers(settings, monkeypatch):
    monkeypatch.setattr(Settings, "secret",
                        classmethod(lambda cls, n: "sk-ant-not-a-real-key"))
    settings.llm_provider = "claude"
    names = [b.name for b in registry.build_llm(settings).backends]
    assert names == ["claude:claude-opus-5", "claude:claude-haiku-4-5"]


def test_the_registry_explains_itself_when_nothing_is_configured(settings,
                                                                monkeypatch):
    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, n: ""))
    settings.llm_provider = "ollama"
    settings.llm_local_endpoint = "http://127.0.0.1:1"
    providers = registry.build(settings, stt_name="typed")
    assert not providers.llm.available
    assert providers.notes and "Claude" in providers.notes[0]


# --------------------------------------------------------------------------
# local tiers (standing routing policy: attempt local first)
# --------------------------------------------------------------------------
def test_the_registry_builds_one_backend_per_registered_tier(settings):
    """The tiers are named in the config so the escalation loop can ask for
    the one it wants; each has to exist as a backend of its own for that."""
    settings.llm_provider = "ollama"
    settings.llm_local_endpoint = "http://127.0.0.1:1"    # nothing listening
    settings.routing_tiers = {"small": "a:1", "mid": "b:2", "json": "c:3"}
    settings.routing_order = ["small", "mid"]
    settings.routing_order_json = ["json", "mid"]

    names = [b.name for b in registry.build_llm(settings).backends]
    assert names[:3] == ["ollama:a:1", "ollama:b:2", "ollama:c:3"]


def test_a_tier_is_ordered_cheapest_first(settings):
    settings.llm_provider = "ollama"
    settings.llm_local_endpoint = "http://127.0.0.1:1"
    settings.routing_tiers = {"json": "c:3", "mid": "b:2", "small": "a:1"}
    settings.routing_order = ["small", "mid"]
    settings.routing_order_json = ["json", "mid"]
    names = [b.name for b in registry.build_llm(settings).backends]
    # Declaration order in the mapping must not decide it; the orders do.
    assert names.index("ollama:a:1") < names.index("ollama:b:2")


def test_a_tier_that_is_not_pulled_says_so_rather_than_substituting(settings):
    """A model named in the config but missing on the machine is a visible
    "not installed", never a quiet answer from a different model."""
    settings.llm_provider = "ollama"
    settings.routing_tiers = {"small": "definitely-not-pulled:0.1b"}
    settings.routing_order = ["small"]
    settings.routing_order_json = []
    backend = registry.build_llm(settings).backends[0]
    assert backend.name == "ollama:definitely-not-pulled:0.1b"
    assert not backend.available
    assert "definitely-not-pulled" in backend.status()


def test_no_tiers_configured_still_builds_the_single_local_model(settings):
    settings.llm_provider = "ollama"
    settings.llm_local_endpoint = "http://127.0.0.1:1"
    settings.routing_tiers = {}
    settings.routing_order = []
    settings.routing_order_json = []
    settings.llm_local_model = "solo:1b"
    names = [b.name for b in registry.build_llm(settings).backends]
    assert "ollama:solo:1b" in names


def test_one_tag_of_a_family_does_not_stand_in_for_another():
    """Found by running it: qwen3:8b reported itself ready because qwen3:4b
    had been pulled, then answered every request with a 404.  Harmless while
    only one Ollama model was ever configured; wrong the moment the routing
    tiers put two tags of one family side by side."""
    from raagacomposer.providers.local_llm import _same_model

    assert _same_model("qwen3:4b", "qwen3:4b")
    assert not _same_model("qwen3:4b", "qwen3:8b")
    assert not _same_model("qwen3:8b", "qwen3:4b")
    # A bare name still means :latest, on either side.
    assert _same_model("llama3:latest", "llama3")
    assert _same_model("llama3", "llama3:latest")
    assert not _same_model("llama3:8b", "llama3")
    assert not _same_model("", "qwen3:4b")
    assert not _same_model("hermes3:8b", "qwen3:8b")


# --------------------------------------------------------------------------
# the judged loop through the router
# --------------------------------------------------------------------------
def test_a_schema_failure_escalates_past_a_confident_local_model(tmp_path):
    """The case that motivated putting schema first: a local model can be
    confident and still return a row with no raaga name in it."""
    from raagacomposer.providers import escalation

    bad = FakeBackend("local", 40, 0.0, local=True)
    bad.suggest_raagas = lambda brief, candidates: [{": ": "raaga"}]
    good = FakeBackend("opus", 90, 15.0, local=False)
    log = escalation.AttemptLog(tmp_path / "attempts.jsonl")
    r = RoutedLLM([lambda: bad, lambda: good], policy="local_first",
                  refresh_seconds=0, attempt_log=log)

    out = r.suggest_raagas(CreativeBrief(), ["K"])
    assert out == [{"raaga": "K", "reason": "opus"}]

    decision = r.last_decision["suggest_raagas"]
    assert [a.verdict for a in decision.attempts] == ["schema", "accepted"]
    assert decision.backend == "opus" and decision.paid and decision.escalated


def test_the_mode_and_the_model_are_recorded_with_every_result():
    """Otherwise runs made under different modes are not comparable, and a
    change we made ourselves looks like the model getting worse."""
    local = FakeBackend("local", 40, 0.0, local=True)
    r = RoutedLLM([lambda: local], policy="local_first", refresh_seconds=0)
    r.suggest_raagas(CreativeBrief(), ["K"])
    decision = r.last_decision["suggest_raagas"]
    assert decision.mode == "local_first"
    assert decision.backend == "local"
    assert r.last_route["suggest_raagas"] == "local"


def test_the_routing_log_keeps_what_each_backend_said(tmp_path):
    from raagacomposer.providers import escalation

    bad = FakeBackend("local", 40, 0.0, local=True)
    bad.suggest_raagas = lambda brief, candidates: [{"raaga": "not-in-the-list"}]
    good = FakeBackend("opus", 90, 15.0, local=False)
    path = tmp_path / "attempts.jsonl"
    r = RoutedLLM([lambda: bad, lambda: good], policy="local_first",
                  refresh_seconds=0, attempt_log=escalation.AttemptLog(path))
    r.suggest_raagas(CreativeBrief(mood="sad"), ["K"])

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["mode"] == "local_first"
    assert row["answered_by"] == "opus"
    assert "not-in-the-list" in row["outputs"]["local"]
    assert "reason" in row["outputs"]["opus"]
    assert row["attempts"][0]["verdict"] == "schema"


def test_an_unjudged_mode_keeps_the_old_accept_anything_behaviour():
    """auto has no thresholds worth applying, so a non-empty answer is
    accepted exactly as it always was - including a malformed one."""
    bad = FakeBackend("opus", 90, 15.0, local=False)
    bad.suggest_raagas = lambda brief, candidates: [{": ": "raaga"}]
    r = RoutedLLM([lambda: bad], policy="auto", refresh_seconds=0)
    assert r.suggest_raagas(CreativeBrief(), ["K"]) == [{": ": "raaga"}]


def test_the_configured_order_decides_the_local_chain_not_strength():
    """Every local model costs nothing, so the cost key cannot separate them
    and ordering by strength put the largest and slowest first - the opposite
    of "a cheap first attempt", and it made the configured order decorative."""
    small = FakeBackend("ollama:small", 40, 0.0, local=True)
    small.model = "small:1b"
    mid = FakeBackend("ollama:mid", 55, 0.0, local=True)
    mid.model = "mid:8b"
    r = RoutedLLM([lambda: mid, lambda: small], policy="local_first",
                  refresh_seconds=0,
                  tiers={"small": "small:1b", "mid": "mid:8b"},
                  order=["small", "mid"])
    assert [b.name for b in r.chain(tasks.WRITE_LYRICS)] == ["ollama:small",
                                                             "ollama:mid"]


def test_a_paid_backend_stays_behind_every_local_rung():
    small = FakeBackend("ollama:small", 40, 0.0, local=True)
    small.model = "small:1b"
    opus = FakeBackend("opus", 90, 15.0, local=False)
    r = RoutedLLM([lambda: opus, lambda: small], policy="local_first",
                  refresh_seconds=0, tiers={"small": "small:1b"},
                  order=["small"])
    assert [b.name for b in r.chain(tasks.WRITE_LYRICS)] == ["ollama:small",
                                                             "opus"]


def test_a_local_backend_the_config_does_not_name_goes_after_the_ones_it_does():
    named = FakeBackend("ollama:named", 40, 0.0, local=True)
    named.model = "small:1b"
    stray = FakeBackend("ollama:stray", 80, 0.0, local=True)
    stray.model = "unlisted:70b"
    r = RoutedLLM([lambda: stray, lambda: named], policy="local_first",
                  refresh_seconds=0, tiers={"small": "small:1b"},
                  order=["small"])
    assert [b.name for b in r.chain(tasks.WRITE_LYRICS)] == ["ollama:named",
                                                             "ollama:stray"]


def test_an_unjudged_mode_keeps_the_strength_ordering():
    small = FakeBackend("ollama:small", 40, 0.0, local=True)
    small.model = "small:1b"
    mid = FakeBackend("ollama:mid", 55, 0.0, local=True)
    mid.model = "mid:8b"
    r = RoutedLLM([lambda: small, lambda: mid], policy="auto",
                  refresh_seconds=0,
                  tiers={"small": "small:1b", "mid": "mid:8b"},
                  order=["small", "mid"])
    # auto has no judge, so the strongest backend still leads a hard task.
    assert r.chain(tasks.WRITE_LYRICS)[0].name == "ollama:mid"
