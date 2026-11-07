"""A funky groove with a raaga inside it (Krish, 7 Nov 2026).

Krish sent a short, catchy clip and asked to build on it instead of the slow two-line sketch. The clip's own melody belongs to its
maker, so it is not copied here. What was taken from it is the *feel*, measured from the audio:

* 140 bpm, a kick on beats 1 and 3, a clap on every beat with ghost hits before 2 and 4, hats on every beat;
* a pumping bass on the tonic and the fifth;
* a hook whose colour, heard over its tonic, is Sa R2 G2 M2 Pa D1 N2 - the raaga Shanmukhapriya.

On that groove this module writes an ORIGINAL call-and-response hook in Shanmukhapriya, a bass line, drums and a little kanjira,
and mixes them through the composer's own synthesiser.  At 140 bpm in four, two bars are one cycle of Adi tala.

Everything is data (step, length, swara) on a sixteen-step bar, so the band's agents can swap a part - "Kali, give me a new
hook" - without touching the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import dsp
from ..core.models import Note
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import Raaga, library

BPM = 140
STEPS = 16                                   # sixteenth notes in a bar of four beats
TONIC = 62                                   # D, where the clip sits

#: (step, length in steps, swara) - two bars of call, two of answer.  Written here, not transcribed.
HOOK: tuple[tuple[tuple[int, int, str], ...], ...] = (
    ((0, 2, "P"), (3, 1, "P"), (4, 2, "D1"), (6, 2, "P"), (8, 1, "M2"), (10, 2, "G2"), (12, 4, "R2")),
    ((0, 2, "G2"), (2, 1, "M2"), (3, 1, "P"), (6, 2, "N2"), (8, 3, "S+"), (12, 2, "N2"), (14, 2, "D1")),
    ((0, 2, "P"), (3, 1, "P"), (4, 2, "D1"), (6, 2, "P"), (8, 1, "S+"), (10, 2, "N2"), (12, 4, "D1")),
    ((0, 2, "P"), (2, 2, "M2"), (4, 2, "G2"), (6, 2, "R2"), (8, 6, "S")),
)

#: The bass: (step, swara, octave shift) per bar, root on the beat, octave and fifth on the off-beats.
BASS_ROOTS = ("S", "S", "G2", "P")
BASS_FIGURE = ((0, 0, 0), (3, 0, 12), (6, 1, 0), (8, 0, 0), (11, 0, 12), (14, 1, 0))   # (step, 0=root 1=fifth-of-root, octave)

#: Drums on the sixteen steps, from the clip: kick 1 and 3 (and a push on the "a" of 3), clap on 2 and 4 with ghosts, hats.
KICK = {0: 118, 8: 112, 11: 70}
CLAP = {4: 112, 12: 112, 6: 46, 14: 46}
HATS = {s: (84 if s % 4 == 0 else 56) for s in range(0, STEPS, 2)}


@dataclass
class Groove:
    bpm: int = BPM
    tonic: int = TONIC
    raaga: str = "Shanmukhapriya"
    intro_bars: int = 2
    hook_repeats: int = 2
    outro_bars: int = 1
    parts: dict[str, list[Note]] = field(default_factory=dict)

    @property
    def step_seconds(self) -> float:
        return 60.0 / self.bpm / 4

    @property
    def bars(self) -> int:
        return self.intro_bars + len(HOOK) * self.hook_repeats + self.outro_bars

    @property
    def seconds(self) -> float:
        return self.bars * STEPS * self.step_seconds


def _note(raaga: Raaga, swara: str, tonic: int, start: float, dur: float, vel: int, shift: int = 0) -> Note:
    return Note(swara=swara, midi=raaga.midi(swara, tonic) + shift, start=round(start, 4), duration=round(dur, 4), velocity=vel)


def build(groove: Groove | None = None) -> Groove:
    """Lay every part out as notes.  Pure: no audio."""
    g = groove or Groove()
    raaga = library().require(g.raaga)
    st = g.step_seconds
    bar_s = STEPS * st
    kit: list[Note] = []
    bass: list[Note] = []
    hook: list[Note] = []
    kanjira: list[Note] = []
    pad: list[Note] = []
    for bar in range(g.bars):
        t0 = bar * bar_s
        last = bar == g.bars - 1
        drums = {0: 120} if last else KICK
        for s, v in drums.items():
            kit.append(Note(swara="-", midi=36, start=t0 + s * st, duration=st * 2, velocity=v))
        if not last:
            for s, v in CLAP.items():
                kit.append(Note(swara="-", midi=37, start=t0 + s * st, duration=st, velocity=v))
            for s, v in HATS.items():
                kit.append(Note(swara="-", midi=38, start=t0 + s * st, duration=st * 0.5, velocity=v))
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        if last:
            bass.append(_note(raaga, "S", g.tonic - 24, t0, bar_s * 0.9, 110))
        else:
            for s, kind, octv in BASS_FIGURE:
                sw = root if kind == 0 else "P" if root in ("S", "G2") else "S+"
                bass.append(_note(raaga, sw, g.tonic - 24, t0 + s * st, st * 1.6, 104 if s % 8 == 0 else 86, octv))
        hook_bar = bar - g.intro_bars
        if 0 <= hook_bar < len(HOOK) * g.hook_repeats:
            phrase = HOOK[hook_bar % len(HOOK)]
            lift = 12 if hook_bar >= len(HOOK) and hook_bar % len(HOOK) in (2,) else 0   # second time round, the third bar jumps up
            for s, length, sw in phrase:
                hook.append(_note(raaga, sw, g.tonic, t0 + s * st, length * st * 0.92, 100, lift))
            if hook_bar >= len(HOOK):                                                     # the kanjira joins the second time
                for s in range(STEPS):
                    kanjira.append(Note(swara="-", midi=48 + (0 if s % 4 == 0 else 2), start=t0 + s * st,
                                        duration=st * 0.5, velocity=70 if s % 4 == 0 else 38))
        if not last:
            pad.append(_note(raaga, "S", g.tonic - 12, t0, bar_s, 40))
            pad.append(_note(raaga, "P", g.tonic - 12, t0, bar_s, 34))
    g.parts = {"drum_kit": kit, "bass": bass, "electric_piano": hook, "kanjira": kanjira, "synth_pad": pad}
    return g


#: The entrance (Krish, 7 Nov 2026): a young artist the crowd is booing before he reaches the stage; ten seconds of silence, then
#: ten seconds to silence the boos - after that the crowd only whistles.  Six bars at 140 bpm is 10.3 s:
#:   bars 1-3  stop-time stabs, each bar a step higher, the kick under every stab and nothing between them;
#:   bars 4-5  a snare-and-kanjira roll from a whisper to a shout under a two-octave run up the arohanam;
#:   bar 6     one huge hit, a beat of dead air, and the hook's last phrase as the pick-up into the groove.
STABS = ((0, ("S", "G2", "P")), (6, ("R2", "M2", "D1")), (10, ("G2", "P", "N2")))
RUN = ("S", "R2", "G2", "M2", "P", "D1", "N2", "S+", "R2+", "G2+", "M2+", "P+", "D1+", "N2+", "S++")
ENTRANCE_BARS = 6


def _shift_token(token: str, octaves: int) -> str:
    return token + "+" * octaves


def entrance(g: Groove, raaga: Raaga) -> dict[str, list[Note]]:
    st = g.step_seconds
    bar_s = STEPS * st
    parts: dict[str, list[Note]] = {k: [] for k in MIX}
    for bar in range(3):                                           # stop-time stabs, climbing
        t0 = bar * bar_s
        for s, chord in STABS:
            v = 96 + 10 * bar
            parts["drum_kit"].append(Note(swara="-", midi=36, start=t0 + s * st, duration=st * 2, velocity=min(127, v + 10)))
            parts["drum_kit"].append(Note(swara="-", midi=37, start=t0 + s * st, duration=st, velocity=v))
            for i, sw in enumerate(chord):
                parts["electric_piano"].append(_note(raaga, sw, g.tonic, t0 + s * st, st * 2.5, v, 2 * bar if i == 0 else 0))
            parts["bass"].append(_note(raaga, chord[0], g.tonic - 24, t0 + s * st, st * 2, 112))
    t0 = 3 * bar_s                                                 # the roll and the run
    for s in range(2 * STEPS):
        v = int(30 + 95 * s / (2 * STEPS - 1))
        parts["drum_kit"].append(Note(swara="-", midi=37, start=t0 + s * st, duration=st * 0.6, velocity=v))
        parts["kanjira"].append(Note(swara="-", midi=50, start=t0 + s * st, duration=st * 0.5, velocity=max(20, v - 20)))
        if s % 4 == 0:
            parts["drum_kit"].append(Note(swara="-", midi=36, start=t0 + s * st, duration=st * 2, velocity=min(127, 60 + v // 2)))
    for i, sw in enumerate(RUN):
        parts["electric_piano"].append(_note(raaga, sw, g.tonic - 12, t0 + i * 2 * st, st * 1.9, 70 + 3 * i))
    t0 = 5 * bar_s                                                 # the hit, the dead air, the pick-up
    parts["drum_kit"] += [Note(swara="-", midi=36, start=t0, duration=st * 4, velocity=127),
                          Note(swara="-", midi=38, start=t0, duration=st * 6, velocity=127),
                          Note(swara="-", midi=37, start=t0, duration=st * 2, velocity=120)]
    parts["bass"].append(_note(raaga, "S", g.tonic - 24, t0, st * 4, 127))
    for sw in ("S", "P", "S+"):
        parts["electric_piano"].append(_note(raaga, sw, g.tonic, t0, st * 4, 124))
    for s, length, sw in ((8, 2, "P"), (10, 2, "M2"), (12, 2, "G2"), (14, 2, "R2")):
        parts["electric_piano"].append(_note(raaga, sw, g.tonic, t0 + s * st, length * st * 0.9, 118))
        parts["drum_kit"].append(Note(swara="-", midi=37, start=t0 + s * st, duration=st, velocity=90 + s * 2))
    return parts


def build_show(silence: float = 10.0, groove: Groove | None = None) -> tuple[Groove, dict[str, list[Note]], float]:
    """Silence, the entrance, then the groove (without its own intro).  -> (groove, parts, total seconds of music)."""
    g = groove or Groove(intro_bars=0, hook_repeats=2)
    raaga = library().require(g.raaga)
    build(g)
    ent = entrance(g, raaga)
    lead_in = silence + ENTRANCE_BARS * STEPS * g.step_seconds
    parts: dict[str, list[Note]] = {k: [] for k in MIX}
    for key in MIX:
        for n in ent.get(key, []):
            parts[key].append(Note(swara=n.swara, midi=n.midi, start=round(n.start + silence, 4), duration=n.duration,
                                   velocity=n.velocity))
        for n in g.parts.get(key, []):
            parts[key].append(Note(swara=n.swara, midi=n.midi, start=round(n.start + lead_in, 4), duration=n.duration,
                                   velocity=n.velocity))
    return g, parts, lead_in + g.seconds


def render_show(silence: float = 10.0, sr: int = 44100, seed: int = 7) -> tuple[np.ndarray, float]:
    """The whole entrance as audio.  -> (audio, seconds at which the entrance starts)."""
    _, parts, seconds = build_show(silence)
    return _mix(parts, seconds + 1.5, sr, seed), silence


def _mix(parts: dict[str, list[Note]], total: float, sr: int, seed: int) -> np.ndarray:
    mix = dsp.silence(total, sr)
    for key, notes in parts.items():
        if not notes:
            continue
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        audio = render_notes(notes, inst, sr, total_seconds=total, seed=seed)
        gain, pan = MIX[key]
        mix = dsp.mix_into(mix, dsp.pan_mono(audio * gain, pan), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.0, sr).astype(np.float32)


#: Level and pan per part.
MIX = {"drum_kit": (0.95, 0.0), "bass": (0.85, 0.0), "electric_piano": (0.8, 0.12), "kanjira": (0.35, -0.3), "synth_pad": (0.25, 0.0)}


def render(groove: Groove | None = None, sr: int = 44100, seed: int = 7) -> tuple[np.ndarray, Groove]:
    g = build(groove)
    total = g.seconds + 1.5
    mix = dsp.silence(total, sr)
    for key, notes in g.parts.items():
        if not notes:
            continue
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        audio = render_notes(notes, inst, sr, total_seconds=total, seed=seed)
        gain, pan = MIX[key]
        mix = dsp.mix_into(mix, dsp.pan_mono(audio * gain, pan), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.01, 1.0, sr).astype(np.float32), g
