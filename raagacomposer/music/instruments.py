"""Instrument catalog (spec section 7).

Two things matter here.  First, when the creator names an instrument it must be
*that* instrument or an honest report that it is unavailable -- never a silent
substitution.  Second, when the creator describes only a feel, the catalog has
to be searchable by that feel.

Each entry carries both identity (names, aliases, family, range, roles) and the
synthesis parameters used by :mod:`raagacomposer.music.synth`.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Instrument:
    key: str
    name: str
    family: str
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    midi_low: int = 48
    midi_high: int = 84
    roles: List[str] = field(default_factory=lambda: ["lead"])
    default_role: str = "lead"
    # synthesis
    harmonics: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.12])
    attack: float = 0.02
    decay: float = 0.15
    sustain: float = 0.7
    release: float = 0.25
    pluck: bool = False
    pluck_decay: float = 2.5
    noise: float = 0.0
    noise_color: float = 3000.0
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.0
    detune: float = 0.0
    voices: int = 1
    inharmonicity: float = 0.0
    brightness: float = 1.0
    percussive: bool = False
    hit_freqs: List[float] = field(default_factory=list)
    hit_decays: List[float] = field(default_factory=list)
    default_gain: float = 0.8
    default_pan: float = 0.0
    glide: float = 0.0
    description: str = ""

    def supports(self, role: str) -> bool:
        return role in self.roles

    def fits(self, midi: int) -> bool:
        return self.midi_low <= midi <= self.midi_high

    @property
    def label(self) -> str:
        return self.name


def _i(**kw) -> Instrument:
    return Instrument(**kw)


CATALOG: List[Instrument] = [
    _i(key="veena", name="Veena", family="plucked-string",
       aliases=["saraswati veena", "vina", "veenai"],
       tags=["classical", "devotional", "warm", "meditative", "intimate", "carnatic",
             "reflective", "traditional"],
       midi_low=41, midi_high=79, roles=["lead", "counter", "fill"],
       harmonics=[1.0, 0.62, 0.4, 0.28, 0.16, 0.09, 0.05], attack=0.006,
       decay=0.5, sustain=0.32, release=0.5, pluck=True, pluck_decay=1.7,
       vibrato_rate=5.2, vibrato_depth=0.06, glide=0.35, brightness=0.95,
       description="Plucked, resonant, deeply Carnatic. Carries gamaka beautifully."),
    _i(key="sitar", name="Sitar", family="plucked-string",
       aliases=["sithar"], tags=["classical", "hindustani", "mystical", "meditative",
                                 "sympathetic", "night"],
       midi_low=45, midi_high=81, roles=["lead", "counter", "fill"],
       harmonics=[1.0, 0.7, 0.55, 0.42, 0.3, 0.22, 0.15, 0.1], attack=0.004,
       decay=0.4, sustain=0.28, release=0.6, pluck=True, pluck_decay=1.5,
       vibrato_depth=0.1, glide=0.5, inharmonicity=0.0008, brightness=1.15,
       description="Buzzing sympathetic resonance; strong identity."),
    _i(key="tanpura", name="Tanpura", family="drone",
       aliases=["tambura", "drone"], tags=["drone", "meditative", "devotional",
                                           "still", "classical"],
       midi_low=36, midi_high=64, roles=["drone", "pad"], default_role="drone",
       harmonics=[1.0, 0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12], attack=0.05,
       decay=1.2, sustain=0.5, release=1.5, pluck=True, pluck_decay=0.55,
       default_gain=0.35, description="The bed the whole raaga sits on."),
    _i(key="violin", name="Violin", family="bowed-string",
       aliases=["fiddle"], tags=["emotional", "romantic", "sad", "longing", "lyrical",
                                 "cinematic", "expressive"],
       midi_low=55, midi_high=88, roles=["lead", "counter", "pad", "fill"],
       harmonics=[1.0, 0.72, 0.5, 0.36, 0.24, 0.16, 0.1], attack=0.09, decay=0.12,
       sustain=0.85, release=0.3, noise=0.035, noise_color=5200,
       vibrato_rate=5.6, vibrato_depth=0.14, glide=0.25,
       description="The most expressive melodic voice in the catalog."),
    _i(key="cello", name="Cello", family="bowed-string",
       aliases=["violoncello"], tags=["warm", "sad", "deep", "intimate", "night",
                                      "cinematic", "melancholy"],
       midi_low=36, midi_high=72, roles=["counter", "bass", "pad", "lead"],
       default_role="counter",
       harmonics=[1.0, 0.66, 0.42, 0.28, 0.18, 0.1], attack=0.11, decay=0.15,
       sustain=0.85, release=0.4, noise=0.03, noise_color=3000,
       vibrato_rate=4.8, vibrato_depth=0.12, default_pan=-0.25,
       description="Low warmth under a vocal without crowding it."),
    _i(key="strings", name="String Ensemble", family="bowed-string",
       aliases=["string section", "orchestra strings", "strings section"],
       tags=["lush", "cinematic", "grand", "romantic", "emotional", "sweeping"],
       midi_low=40, midi_high=88, roles=["pad", "counter", "lead", "fill"],
       default_role="pad",
       harmonics=[1.0, 0.7, 0.48, 0.32, 0.2, 0.12], attack=0.22, decay=0.2,
       sustain=0.9, release=0.7, noise=0.02, vibrato_rate=4.6, vibrato_depth=0.08,
       detune=0.11, voices=5, default_gain=0.55,
       description="Wide bed; use behind the voice, not on top of it."),
    _i(key="saxophone", name="Saxophone", family="wind",
       aliases=["sax", "tenor sax", "alto sax"],
       tags=["night", "lonely", "warm", "jazz", "smoky", "urban", "intimate"],
       midi_low=46, midi_high=79, roles=["lead", "counter", "fill"],
       harmonics=[1.0, 0.85, 0.55, 0.45, 0.3, 0.22, 0.14], attack=0.05, decay=0.1,
       sustain=0.85, release=0.25, noise=0.05, noise_color=2400,
       vibrato_rate=5.0, vibrato_depth=0.1, glide=0.2, brightness=1.1,
       description="Late-night warmth; sits between voice and horns."),
    _i(key="flute", name="Bamboo Flute", family="wind",
       aliases=["bansuri", "venu", "bansri", "pullanguzhal"],
       tags=["pastoral", "gentle", "romantic", "devotional", "calm", "innocent",
             "morning", "light"],
       midi_low=60, midi_high=93, roles=["lead", "counter", "fill"],
       harmonics=[1.0, 0.25, 0.14, 0.06], attack=0.06, decay=0.08, sustain=0.9,
       release=0.2, noise=0.07, noise_color=6000, vibrato_rate=5.4,
       vibrato_depth=0.09, glide=0.3,
       description="Breathy and open; the classic counter-voice to a singer."),
    _i(key="shehnai", name="Shehnai", family="wind",
       aliases=["nadaswaram", "nadhaswaram", "shenai"],
       tags=["auspicious", "wedding", "celebration", "ceremonial", "bright",
             "traditional"],
       midi_low=55, midi_high=84, roles=["lead", "counter", "fill"],
       harmonics=[1.0, 0.9, 0.7, 0.6, 0.45, 0.35, 0.25, 0.18], attack=0.04,
       decay=0.1, sustain=0.88, release=0.2, noise=0.04, vibrato_rate=5.8,
       vibrato_depth=0.12, brightness=1.3,
       description="Piercing and ceremonial. Loud by nature; keep it low in the mix."),
    _i(key="clarinet", name="Clarinet", family="wind", aliases=[],
       tags=["woody", "warm", "reflective", "gentle", "nostalgia"],
       midi_low=50, midi_high=84, roles=["lead", "counter", "pad"],
       harmonics=[1.0, 0.05, 0.5, 0.04, 0.28, 0.03, 0.15], attack=0.05,
       decay=0.1, sustain=0.88, release=0.22, noise=0.03, vibrato_depth=0.05,
       description="Hollow odd-harmonic tone; blends under strings."),
    _i(key="piano", name="Piano", family="keyboard",
       aliases=["grand piano", "acoustic piano", "keys"],
       tags=["intimate", "sad", "romantic", "modern", "clear", "reflective",
             "cinematic", "lonely"],
       midi_low=28, midi_high=96, roles=["lead", "pad", "counter", "bass", "fill"],
       harmonics=[1.0, 0.45, 0.3, 0.18, 0.12, 0.07, 0.04], attack=0.004,
       decay=0.6, sustain=0.25, release=0.5, pluck=True, pluck_decay=1.1,
       inharmonicity=0.0004, description="Reliable and neutral; good first pass."),
    _i(key="electric_piano", name="Electric Piano", family="keyboard",
       aliases=["rhodes", "wurlitzer", "e-piano"],
       tags=["warm", "soft", "night", "intimate", "modern", "smooth"],
       midi_low=33, midi_high=91, roles=["pad", "counter", "lead"],
       default_role="pad",
       harmonics=[1.0, 0.3, 0.16, 0.08, 0.04], attack=0.008, decay=0.7,
       sustain=0.4, release=0.6, pluck=True, pluck_decay=0.9, brightness=0.85,
       description="Soft bell-like keys; sits well under a close vocal."),
    _i(key="harmonium", name="Harmonium", family="keyboard",
       aliases=["shruti box"], tags=["devotional", "traditional", "warm", "folk",
                                     "prayerful"],
       midi_low=41, midi_high=84, roles=["pad", "counter", "drone"],
       default_role="pad",
       harmonics=[1.0, 0.75, 0.55, 0.4, 0.3, 0.2, 0.14], attack=0.09, decay=0.1,
       sustain=0.95, release=0.25, noise=0.02, default_gain=0.5,
       description="Reedy sustain; the bhajan sound."),
    _i(key="guitar", name="Acoustic Guitar", family="plucked-string",
       aliases=["acoustic guitar", "nylon guitar", "steel guitar"],
       tags=["folk", "warm", "intimate", "modern", "pastoral", "light"],
       midi_low=40, midi_high=84, roles=["lead", "counter", "pad", "rhythm"],
       harmonics=[1.0, 0.55, 0.35, 0.22, 0.14, 0.08], attack=0.005, decay=0.45,
       sustain=0.3, release=0.4, pluck=True, pluck_decay=2.0,
       description="Fingerpicked warmth; good for verses."),
    _i(key="electric_guitar", name="Electric Guitar", family="plucked-string",
       aliases=["e-guitar", "lead guitar"],
       tags=["modern", "energetic", "urban", "intense", "cinematic"],
       midi_low=40, midi_high=88, roles=["lead", "counter", "rhythm"],
       harmonics=[1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2], attack=0.01, decay=0.3,
       sustain=0.6, release=0.35, vibrato_depth=0.08, brightness=1.2, glide=0.2,
       description="Sustained and forward; will fight the vocal if left loud."),
    _i(key="bass", name="Bass Guitar", family="bass",
       aliases=["bass guitar", "electric bass", "low end"],
       tags=["modern", "groove", "foundation", "urban"],
       midi_low=28, midi_high=60, roles=["bass"], default_role="bass",
       harmonics=[1.0, 0.4, 0.18, 0.08], attack=0.008, decay=0.4, sustain=0.5,
       release=0.3, pluck=True, pluck_decay=1.4, default_gain=0.7,
       description="Root movement; keep it below the vocal fundamental."),
    _i(key="double_bass", name="Double Bass", family="bass",
       aliases=["upright bass", "contrabass"],
       tags=["warm", "jazz", "night", "acoustic", "smoky"],
       midi_low=28, midi_high=60, roles=["bass"], default_role="bass",
       harmonics=[1.0, 0.5, 0.25, 0.12], attack=0.03, decay=0.3, sustain=0.6,
       release=0.4, noise=0.02, default_gain=0.65,
       description="Woody low end for intimate arrangements."),
    _i(key="santoor", name="Santoor", family="struck-string",
       aliases=["santur"], tags=["shimmer", "mystical", "gentle", "night", "calm",
                                 "reflective"],
       midi_low=48, midi_high=91, roles=["counter", "fill", "pad", "lead"],
       default_role="counter",
       harmonics=[1.0, 0.6, 0.42, 0.3, 0.22, 0.15, 0.1], attack=0.003, decay=0.35,
       sustain=0.2, release=0.6, pluck=True, pluck_decay=2.6,
       inharmonicity=0.0006, brightness=1.2,
       description="Rain-like shimmer; excellent for interludes."),
    _i(key="sarod", name="Sarod", family="plucked-string", aliases=[],
       tags=["classical", "hindustani", "deep", "meditative", "expressive"],
       midi_low=41, midi_high=79, roles=["lead", "counter"],
       harmonics=[1.0, 0.68, 0.5, 0.36, 0.24, 0.16], attack=0.005, decay=0.45,
       sustain=0.3, release=0.5, pluck=True, pluck_decay=1.9, glide=0.55,
       description="Fretless glide; heavy meend between notes."),
    _i(key="choir", name="Choir", family="voice",
       aliases=["chorus voices", "backing vocals", "humming"],
       tags=["grand", "devotional", "cinematic", "lush", "sacred", "emotional"],
       midi_low=45, midi_high=81, roles=["pad", "counter"], default_role="pad",
       harmonics=[1.0, 0.6, 0.35, 0.2, 0.1], attack=0.35, decay=0.2, sustain=0.9,
       release=0.9, noise=0.03, detune=0.14, voices=6, default_gain=0.45,
       vibrato_rate=4.4, vibrato_depth=0.07,
       description="Wordless voices; adds scale without adding notes."),
    _i(key="synth_pad", name="Synth Pad", family="synth",
       aliases=["pad", "ambient pad", "atmosphere"],
       tags=["modern", "ambient", "cinematic", "night", "spacious", "cold"],
       midi_low=33, midi_high=91, roles=["pad", "drone"], default_role="pad",
       harmonics=[1.0, 0.5, 0.33, 0.25, 0.18, 0.12, 0.08], attack=0.6, decay=0.3,
       sustain=0.95, release=1.4, detune=0.09, voices=3, default_gain=0.4,
       brightness=0.8, description="Wash that fills space between real instruments."),
    _i(key="mridangam", name="Mridangam", family="percussion",
       aliases=["mridangam drum", "mrudangam"],
       tags=["carnatic", "classical", "rhythm", "traditional", "energetic"],
       midi_low=36, midi_high=60, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[92.0, 140.0, 220.0, 330.0],
       hit_decays=[0.35, 0.22, 0.14, 0.09], noise=0.35, default_gain=0.7,
       description="The Carnatic rhythmic backbone."),
    _i(key="tabla", name="Tabla", family="percussion",
       aliases=["tabla set"], tags=["hindustani", "rhythm", "classical", "light",
                                    "traditional"],
       midi_low=36, midi_high=64, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[110.0, 196.0, 294.0, 440.0],
       hit_decays=[0.3, 0.18, 0.12, 0.07], noise=0.22, default_gain=0.65,
       description="Crisp and articulate; lighter than mridangam."),
    _i(key="ghatam", name="Ghatam", family="percussion",
       aliases=["clay pot"], tags=["carnatic", "rhythm", "earthy", "folk"],
       midi_low=36, midi_high=60, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[130.0, 260.0, 520.0],
       hit_decays=[0.18, 0.1, 0.05], noise=0.45, default_gain=0.55,
       description="Dry clay resonance; pairs with mridangam."),
    _i(key="kanjira", name="Kanjira", family="percussion",
       aliases=["frame drum"], tags=["carnatic", "rhythm", "bright", "folk"],
       midi_low=48, midi_high=72, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[220.0, 440.0, 880.0],
       hit_decays=[0.12, 0.07, 0.04], noise=0.6, default_gain=0.45,
       description="High jingling frame drum for lift."),
    _i(key="drum_kit", name="Drum Kit", family="percussion",
       aliases=["drums", "kit", "acoustic drums"],
       tags=["modern", "energetic", "urban", "groove", "pop"],
       midi_low=35, midi_high=60, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[60.0, 180.0, 3200.0],
       hit_decays=[0.28, 0.14, 0.05], noise=0.5, default_gain=0.7,
       description="Kick, snare and hats for a modern arrangement."),
    _i(key="tambourine", name="Tambourine", family="percussion",
       aliases=["shaker", "jingles"], tags=["light", "bright", "folk", "celebration"],
       midi_low=60, midi_high=84, roles=["rhythm"], default_role="rhythm",
       percussive=True, hit_freqs=[2800.0, 5200.0, 8000.0],
       hit_decays=[0.06, 0.04, 0.03], noise=0.85, default_gain=0.35,
       description="Top-end sparkle on the offbeat."),
]

BY_KEY: Dict[str, Instrument] = {i.key: i for i in CATALOG}


def all_instruments() -> List[Instrument]:
    return list(CATALOG)


def keys() -> List[str]:
    return [i.key for i in CATALOG]


def get(key: str) -> Optional[Instrument]:
    return BY_KEY.get((key or "").strip().lower())


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("-", " ").split())


def find(text: str) -> Optional[Instrument]:
    """Exact-ish lookup by name or alias. Returns None rather than guessing."""
    q = _norm(text)
    if not q:
        return None
    for inst in CATALOG:
        names = [inst.key, _norm(inst.name)] + [_norm(a) for a in inst.aliases]
        if q in names:
            return inst
    for inst in CATALOG:
        names = [inst.key.replace("_", " "), _norm(inst.name)] + \
                [_norm(a) for a in inst.aliases]
        for n in names:
            if n and (q == n or q.rstrip("s") == n.rstrip("s")):
                return inst
    return None


def find_in_text(text: str) -> Optional[Instrument]:
    """Longest instrument name mentioned anywhere inside a spoken sentence."""
    blob = _norm(text)
    best: Optional[Instrument] = None
    best_len = 0
    for inst in CATALOG:
        candidates = [inst.key.replace("_", " "), _norm(inst.name)] + \
                     [_norm(a) for a in inst.aliases]
        for c in candidates:
            if not c:
                continue
            if c in blob and len(c) > best_len:
                best_len, best = len(c), inst
    return best


def closest(text: str, n: int = 3) -> List[Instrument]:
    """Nearest catalog entries when an exact instrument is not available."""
    q = _norm(text)
    pool: Dict[str, Instrument] = {}
    for inst in CATALOG:
        pool[_norm(inst.name)] = inst
        pool[inst.key.replace("_", " ")] = inst
        for a in inst.aliases:
            pool[_norm(a)] = inst
    matches = difflib.get_close_matches(q, list(pool), n=n * 2, cutoff=0.4)
    out: List[Instrument] = []
    for m in matches:
        inst = pool[m]
        if inst not in out:
            out.append(inst)
    return out[:n]


def suggest_for_feel(words: Sequence[str], avoid: Sequence[str] = (),
                     role: str = "", limit: int = 4) -> List[Tuple[Instrument, float]]:
    """Rank instruments against a set of feel words (spec section 7.2)."""
    words = [w.lower() for w in words if w]
    avoid_keys = set()
    for a in avoid:
        inst = find(a) or find_in_text(a)
        if inst:
            avoid_keys.add(inst.key)
    scored: List[Tuple[Instrument, float]] = []
    for inst in CATALOG:
        if inst.key in avoid_keys:
            continue
        if role and not inst.supports(role):
            continue
        hits = sum(1.0 for t in inst.tags if any(t in w or w in t for w in words))
        if hits <= 0:
            continue
        scored.append((inst, hits))
    scored.sort(key=lambda s: (-s[1], s[0].name))
    return scored[:limit]


def role_default(inst: Instrument, section_kind: str = "") -> str:
    if inst.percussive:
        return "rhythm"
    if section_kind in ("prelude", "interlude", "bridge", "outro"):
        return "lead" if inst.supports("lead") else inst.default_role
    return inst.default_role


def describe(inst: Instrument) -> str:
    return (f"{inst.name} ({inst.family})\n"
            f"  Range: MIDI {inst.midi_low}-{inst.midi_high}\n"
            f"  Roles: {', '.join(inst.roles)}\n"
            f"  Feel:  {', '.join(inst.tags)}\n"
            f"  {inst.description}")
