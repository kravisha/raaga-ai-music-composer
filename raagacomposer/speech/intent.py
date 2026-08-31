"""Intent and command interpreter (spec sections 5, 7).

Deterministic rules first: they are fast enough to run on every partial
transcript, they never invent an instrument the creator did not ask for, and
they work with no credentials.  A language model is consulted only for
sentences the rules cannot classify, and its answer is still mapped onto the
same closed set of intents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging_setup import get_logger
from ..music import instruments as inst_catalog
from ..raaga.library import library as raaga_library
from ..raaga.selection import expand_feel_words
from .timeline_parser import TimeContext, TimeSpec, describe as describe_time
from .timeline_parser import parse as parse_time

log = get_logger("intent")

INTENTS = (
    "transport.play", "transport.pause", "transport.stop", "transport.resume",
    "transport.seek", "transport.loop",
    "arrange.add", "arrange.remove", "arrange.replace", "arrange.mute",
    "arrange.solo", "arrange.level", "arrange.regenerate", "arrange.suggest",
    "arrange.auto",
    "tune.generate", "tune.variation", "tune.accept", "tune.regenerate_section",
    "tune.tempo",
    "lyrics.generate", "lyrics.accept",
    "raaga.set", "raaga.suggest", "raaga.lock",
    "voice.set", "voice.direction", "voice.render", "voice.vocal_only",
    "mix.full", "mix.instrumental", "mix.export",
    "region.lock", "region.unlock",
    "project.save", "project.undo", "project.redo", "project.cancel",
    "agent.learn", "agent.explain", "agent.feedback", "agent.status",
    "unknown",
)

STOP_WORDS = ("stop", "halt", "cancel that", "never mind", "quiet")


@dataclass
class Command:
    intent: str = "unknown"
    text: str = ""
    time: Optional[TimeSpec] = None
    instrument: str = ""
    target_instrument: str = ""
    section_id: str = ""
    raaga: str = ""
    value: Optional[float] = None
    feel_words: List[str] = field(default_factory=list)
    style: str = ""
    confidence: float = 0.0
    interpretation: str = ""
    source: str = "rules"
    raw_slots: Dict[str, Any] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.intent != "unknown"

    def target_key(self) -> str:
        """Job target for supersession: same target means newest wins."""
        if self.intent.startswith("transport"):
            return "playback"
        if self.intent.startswith("arrange"):
            span = ""
            if self.time and self.time.start is not None:
                span = f"{self.time.start:.0f}-{self.time.end or -1:.0f}"
            return f"arrangement:{self.instrument or 'any'}:{span}"
        if self.intent.startswith("tune"):
            return f"melody:{self.section_id or 'all'}"
        if self.intent.startswith("lyrics"):
            return "lyrics"
        if self.intent.startswith("voice"):
            return "vocal"
        if self.intent.startswith("mix"):
            return "mix"
        if self.intent.startswith("agent"):
            return "agent"
        return self.intent


# --------------------------------------------------------------------------
# rule tables
# --------------------------------------------------------------------------
_RULES: List[Tuple[str, str]] = [
    # The learning agent, checked first so "study this raaga" is not read as
    # an arrangement instruction.
    (r"\b(does not sound like|doesn't sound like|does not sound|"
     r"doesn't sound|too mechanical|sounds wrong|that is wrong|"
     r"i like the|keep the (first|second|third)|that was (good|lovely|nice))\b",
     "agent.feedback"),
    (r"\b(learn|study|practise|practice)\b", "agent.learn"),
    (r"\b(why did you|why do you|explain|how do you know|what do you know|"
     r"where did you learn|what have you learned)\b", "agent.explain"),
    (r"\b(how is your learning|learning progress|what are you learning|"
     r"how much have you learned|what are you studying)\b", "agent.status"),
    # transport
    (r"\b(pause|hold on|wait)\b", "transport.pause"),
    (r"\b(stop|halt)\b(?!.*\b(drum|instrument|violin|adding)\b)", "transport.stop"),
    (r"\b(continue|resume|carry on|keep going|go on)\b", "transport.resume"),
    (r"\b(loop|repeat that|again and again)\b", "transport.loop"),
    (r"\b(play|listen to|hear|let me hear|start playback|playback)\b", "transport.play"),
    (r"\b(go back|rewind|skip (forward|ahead)|jump to|seek)\b", "transport.seek"),
    # arrangement
    (r"\b(replace|swap|change)\b.*\b(with|for|to)\b", "arrange.replace"),
    (r"\b(add|bring in|bring|put|include|layer|introduce|use)\b", "arrange.add"),
    (r"\b(take out|remove|drop|delete|lose|get rid of|mute out|no more)\b",
     "arrange.remove"),
    (r"\btake\b[^.]*\bout\b", "arrange.remove"),
    (r"\b(mute|silence that)\b", "arrange.mute"),
    (r"\bsolo\b", "arrange.solo"),
    (r"\b(louder|quieter|softer|lighter|heavier|gentler|stronger|"
     r"more|less)\b", "arrange.level"),
    # "turn the violin up", "turn it down" - the object sits in the middle.
    (r"\bturn\b[^.]*\b(up|down)\b", "arrange.level"),
    (r"\b(regenerate|redo|try again|another take|different take)\b",
     "arrange.regenerate"),
    (r"\b(suggest|what would|something that fits|another instrument|"
     r"which instrument|pick an instrument|give me an instrument)\b",
     "arrange.suggest"),
    (r"\b(arrange|full arrangement|build the arrangement|orchestrate)\b",
     "arrange.auto"),
    # tune
    (r"\b(variation|another version of the tune|vary the tune)\b", "tune.variation"),
    (r"\b(generate|write|make|compose|give me)\b.*\b(tune|melody)\b",
     "tune.generate"),
    (r"\b(regenerate|rewrite|change)\b.*\b(section|pallavi|charanam|verse|"
     r"chorus|interlude|prelude|outro|bridge)\b", "tune.regenerate_section"),
    (r"\b(accept|approve|lock)\b.*\b(tune|melody)\b", "tune.accept"),
    (r"\b(tempo|bpm|faster|slower|speed)\b", "tune.tempo"),
    # lyrics
    (r"\b(lyrics|words|write the words)\b", "lyrics.generate"),
    # raaga
    (r"\b(raaga|raagam|scale)\b.*\b(suggest|options|alternatives|what)\b",
     "raaga.suggest"),
    (r"\b(use|set|switch to|change to)\b.*\b(raaga|raagam)\b", "raaga.set"),
    (r"\block\b.*\b(raaga|raagam)\b", "raaga.lock"),
    # voice
    (r"\b(without instruments|vocal only|only the voice|just the voice|"
     r"voice alone|a cappella|acapella)\b", "voice.vocal_only"),
    (r"\b(sing|render the vocal|vocal preview|hear the voice)\b", "voice.render"),
    (r"\b(singer|voice)\b.*\b(change|use|switch|set)\b", "voice.set"),
    (r"\b(sing it|make it) (softer|stronger|sadder|happier|more emotional)\b",
     "voice.direction"),
    # mix and export
    (r"\b(instrumental|karaoke|without the (voice|vocal))\b", "mix.instrumental"),
    (r"\b(full mix|mix it|final mix|mix the song|the whole mix)\b", "mix.full"),
    (r"\b(export|render to file|bounce|save as (wav|mp3))\b", "mix.export"),
    # locking and project
    (r"\b(lock)\b", "region.lock"),
    (r"\b(unlock)\b", "region.unlock"),
    (r"\b(save)\b", "project.save"),
    (r"\b(undo)\b", "project.undo"),
    (r"\b(redo)\b", "project.redo"),
    (r"\b(cancel|abort|forget that|never mind)\b", "project.cancel"),
]

LEVEL_WORDS = {
    "louder": 1.25, "turn it up": 1.25, "turn up": 1.25, "more": 1.15,
    "heavier": 1.3, "stronger": 1.2,
    "quieter": 0.8, "softer": 0.78, "lighter": 0.7, "turn it down": 0.8,
    "turn down": 0.8, "less": 0.85, "gentler": 0.75,
}

STYLE_WORDS = ("soft", "intimate", "strong", "emotional", "romantic", "sad",
               "energetic", "devotional", "smooth", "dramatic")


def _clean(text: str) -> str:
    return " " + re.sub(r"\s+", " ", (text or "").lower().strip(" .!?,")) + " "


def interpret(text: str, ctx: TimeContext, llm=None,
              last_instrument: str = "") -> Command:
    """Classify one utterance and pull out its slots."""
    cmd = Command(text=text or "")
    t = _clean(text)
    if not t.strip():
        return cmd

    for pattern, intent in _RULES:
        if re.search(pattern, t):
            cmd.intent = intent
            cmd.confidence = 0.8
            break

    cmd.time = parse_time(text, ctx)
    if cmd.time and cmd.time.section_id:
        cmd.section_id = cmd.time.section_id

    # Instruments: an explicit name always wins over inference.
    named = _instruments_in(t)
    if named:
        cmd.instrument = named[0].key
        if len(named) > 1:
            cmd.target_instrument = named[1].key

    raaga = raaga_library().find_in_text(text)
    if raaga:
        cmd.raaga = raaga.name
        if cmd.intent in ("unknown", "arrange.add"):
            cmd.intent = "raaga.set"
            cmd.confidence = 0.75

    cmd.feel_words = expand_feel_words(text)
    for style in STYLE_WORDS:
        if re.search(rf"\b{style}\b", t):
            cmd.style = style
            break

    # Numeric slots.
    m = re.search(r"\b(\d{2,3})\s*(bpm|beats per minute)\b", t)
    if m:
        cmd.value = float(m.group(1))
        cmd.intent = "tune.tempo"
        cmd.confidence = 0.9
    elif cmd.intent == "tune.tempo":
        if re.search(r"\bfaster|speed (it )?up|quicker\b", t):
            cmd.value = 1.12
        elif re.search(r"\bslower|slow (it )?down\b", t):
            cmd.value = 0.9

    if cmd.intent == "arrange.level":
        for word, factor in LEVEL_WORDS.items():
            if word in t:
                cmd.value = factor
                break
        if cmd.value is None and re.search(r"\bturn\b[^.]*\bup\b", t):
            cmd.value = 1.25
        elif cmd.value is None and re.search(r"\bturn\b[^.]*\bdown\b", t):
            cmd.value = 0.8
        cmd.value = cmd.value or 0.85

    # "replace X with Y" ordering.
    if cmd.intent == "arrange.replace":
        m = re.search(r"\breplace\s+(?:the\s+)?(.+?)\s+(?:with|for|by)\s+(?:the\s+)?(.+)$",
                      t.strip())
        if not m:
            m = re.search(r"\b(?:change|swap)\s+(?:the\s+)?(.+?)\s+(?:to|with|for)\s+"
                          r"(?:the\s+)?(.+)$", t.strip())
        if m:
            a = inst_catalog.find_in_text(m.group(1))
            b = inst_catalog.find_in_text(m.group(2))
            if a:
                cmd.instrument = a.key
            if b:
                cmd.target_instrument = b.key

    # Feel-only instrument request: a described mood with no instrument named.
    if cmd.intent in ("arrange.add", "arrange.suggest") and not cmd.instrument:
        if cmd.feel_words:
            cmd.intent = "arrange.suggest"
            cmd.confidence = max(cmd.confidence, 0.6)
    if (cmd.intent == "unknown" and cmd.feel_words
            and re.search(r"\b(feel|feels|feeling|sound|sounds|mood|vibe)\b", t)):
        cmd.intent = "arrange.suggest"
        cmd.confidence = 0.55

    # Bare "that instrument" style references.
    if (cmd.intent.startswith("arrange") and not cmd.instrument
            and re.search(r"\b(that|the same|it)\b", t) and last_instrument):
        cmd.instrument = last_instrument

    if cmd.intent == "unknown" and llm is not None and getattr(llm, "available", False):
        try:
            guess = llm.classify_intent(text, INTENTS)
            if guess and guess.get("intent") in INTENTS:
                cmd.intent = guess["intent"]
                cmd.confidence = float(guess.get("confidence", 0.5))
                cmd.source = "llm"
                if guess.get("instrument") and not cmd.instrument:
                    found = inst_catalog.find(guess["instrument"])
                    if found:
                        cmd.instrument = found.key
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM intent classification failed: %s", exc)

    cmd.interpretation = describe(cmd)
    return cmd


def _instruments_in(text: str) -> List[inst_catalog.Instrument]:
    """Every catalog instrument mentioned, in the order they appear."""
    found: List[Tuple[int, inst_catalog.Instrument]] = []
    blob = text.lower().replace("-", " ")
    for inst in inst_catalog.all_instruments():
        names = [inst.key.replace("_", " "), inst.name.lower()] + \
                [a.lower() for a in inst.aliases]
        best = None
        for n in names:
            pos = blob.find(n)
            if pos >= 0 and (best is None or pos < best[0] or len(n) > len(best[1])):
                best = (pos, n)
        if best:
            found.append((best[0], inst))
    found.sort(key=lambda p: p[0])
    out: List[inst_catalog.Instrument] = []
    for _, inst in found:
        if inst not in out:
            out.append(inst)
    return out


def unavailable_instrument(text: str) -> Optional[Tuple[str, List[str]]]:
    """Detect an instrument request the catalog cannot satisfy.

    Returns (requested phrase, closest alternatives) so the app can report it
    clearly instead of silently substituting (spec 7.1).
    """
    m = re.search(r"\b(?:add|bring in|bring|use|put|play|include)\s+"
                  r"(?:some|a|an|the)?\s*([a-z][a-z \-]{2,24})", text.lower())
    if not m:
        return None
    phrase = m.group(1).strip()
    phrase = re.split(r"\b(here|there|after|before|from|at|for|in|on|to|and|"
                      r"instead|now|please)\b", phrase)[0].strip()
    if not phrase or inst_catalog.find_in_text(phrase):
        return None
    close = inst_catalog.closest(phrase, 3)
    if not close:
        return None
    return phrase, [c.name for c in close]


def describe(cmd: Command) -> str:
    """One-line, human-readable statement of what the app understood."""
    when = describe_time(cmd.time) if cmd.time else ""
    inst = inst_catalog.get(cmd.instrument)
    target = inst_catalog.get(cmd.target_instrument)
    name = inst.name if inst else cmd.instrument
    target_name = target.name if target else cmd.target_instrument

    if cmd.intent == "transport.play":
        return f"Play {when}" if when else "Play"
    if cmd.intent == "transport.seek" and cmd.time:
        return f"Move the playhead {when}"
    if cmd.intent == "arrange.add":
        return f"Add {name or 'an instrument'}{' ' + when if when else ''}"
    if cmd.intent == "arrange.remove":
        return f"Take {name or 'that instrument'} out{' ' + when if when else ''}"
    if cmd.intent == "arrange.replace":
        return f"Replace {name or '?'} with {target_name or '?'}" + \
            (f" {when}" if when else "")
    if cmd.intent == "arrange.suggest":
        feel = ", ".join(cmd.feel_words[:3])
        return f"Suggest an instrument for: {feel or 'this feel'}" + \
            (f" {when}" if when else "")
    if cmd.intent == "arrange.level":
        direction = "louder" if (cmd.value or 1) > 1 else "softer"
        return f"Make {name or 'that'} {direction}" + (f" {when}" if when else "")
    if cmd.intent == "tune.tempo" and cmd.value:
        return (f"Set the tempo to {cmd.value:.0f} bpm" if cmd.value > 20
                else f"Change the tempo by x{cmd.value:.2f}")
    if cmd.intent == "raaga.set":
        return f"Use raaga {cmd.raaga}"
    if cmd.intent == "voice.vocal_only":
        return "Produce the studio vocal-only master"
    if cmd.intent == "region.lock":
        return f"Lock {when or 'the current selection'}"
    if cmd.intent == "region.unlock":
        return f"Unlock {when or 'the current selection'}"
    if cmd.intent == "unknown":
        return "Not understood"
    pretty = cmd.intent.split(".")[-1].replace("_", " ")
    return f"{pretty.capitalize()}{' ' + when if when else ''}"
