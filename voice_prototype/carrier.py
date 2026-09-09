"""Guide synthesis — contract `5ee63d7` §1.3, §6.

Real recorded vocal samples from a CVVC bank, driven by the tune's **continuous
pitch curve** rather than by one MIDI number per note. That combination is the
point: the samples carry vowel and consonant identity that a formant model
approximates, and the curve carries the ornamentation that a note grid destroys.

Two rules this stage exists to keep:

* **A rest is silent.** The curve is unvoiced there and nothing is synthesised
  into it. Renders are assembled per phrase, and phrases never bridge a rest.
* **Nothing is approximated.** An alias the bank does not hold is a refusal from
  the frontend, before a sample is read.

Reads the bank and the engine where they lie. Writes only where it is told.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100

#: Peak the guide is normalised to. Below full scale on purpose: a render that
#: sits on the rail cannot be repaired by turning it down afterwards.
PEAK_TARGET = 0.89


class CarrierError(Exception):
    """A refusal. Never a silent correction."""


class Request(C.Structure):
    _fields_ = [
        ("sample_fs", C.c_int32), ("sample_length", C.c_int32),
        ("sample", C.POINTER(C.c_double)), ("frq_length", C.c_int32),
        ("frq", C.c_void_p), ("tone", C.c_int32), ("con_vel", C.c_double),
        ("offset", C.c_double), ("required_length", C.c_double),
        ("consonant", C.c_double), ("cut_off", C.c_double), ("volume", C.c_double),
        ("modulation", C.c_double), ("tempo", C.c_double),
        ("pitch_bend_length", C.c_int32), ("pitch_bend", C.c_void_p),
        ("flag_g", C.c_int32), ("flag_O", C.c_int32), ("flag_P", C.c_int32),
        ("flag_Mt", C.c_int32), ("flag_Mb", C.c_int32), ("flag_Mv", C.c_int32)]


_pd = C.POINTER(C.c_double)
_pf = C.POINTER(C.c_float)
_logger_type = C.CFUNCTYPE(None, C.c_char_p)


def read_bank(bank_zip: Path) -> tuple:
    """Alias name to (filename, timing fields)."""
    bank = zipfile.ZipFile(str(bank_zip))
    aliases = {}
    for line in bank.read("oto.ini").decode("utf-8-sig").splitlines():
        if "=" not in line:
            continue
        filename, values = line.split("=", 1)
        alias, *fields = values.split(",")
        aliases.setdefault(alias or Path(filename).stem,
                           (filename, [float(x or 0) for x in fields]))
    return bank, aliases


def render(placed: list, tune, requests: list, *, bank_zip: Path, dll_path: Path,
           destination: Path, breath: float = 0.47, tension: float = 0.46) -> dict:
    """One guide file for the whole tune, phrase by phrase.

    Each contiguous sung stretch is synthesised on its own and placed at its own
    offset in a silent buffer of the tune's full length. The silence between
    phrases is never written into, so a rest is a rest in the output and not
    merely in the record.
    """
    if len(placed) != len(requests):
        raise CarrierError(
            f"{len(placed)} placed syllables against {len(requests)} alias "
            f"requests; these must correspond one to one.")
    destination = Path(destination)
    if destination.exists():
        raise CarrierError(f"{destination} exists; renders are not overwritten.")

    bank, aliases = read_bank(Path(bank_zip))
    total = int(round(tune.duration_s * SAMPLE_RATE))
    if total <= 0:
        raise CarrierError("The tune has no duration.")
    mixed = np.zeros(total, dtype=np.float64)
    events: list = []

    phrases = sorted({item.phrase for item in placed})
    for phrase in phrases:
        members = [(item, request) for item, request in zip(placed, requests)
                   if item.phrase == phrase]
        piece, offset, phrase_events = _render_phrase(
            members, tune, bank, aliases, Path(dll_path), breath, tension)
        begin = int(round(offset * SAMPLE_RATE))
        end = min(total, begin + len(piece))
        if end > begin:
            mixed[begin:end] += piece[:end - begin]
        events += phrase_events

    peak = float(np.max(np.abs(mixed)))
    if peak <= 0:
        raise CarrierError("The guide is silent; nothing was synthesised.")
    mixed *= PEAK_TARGET / peak
    fade = int(0.008 * SAMPLE_RATE)
    mixed[:fade] *= np.linspace(0, 1, fade)
    mixed[-fade:] *= np.linspace(1, 0, fade)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), mixed, SAMPLE_RATE, subtype="PCM_24")

    silent = _rest_levels(mixed, tune)
    return {
        "output": str(destination),
        "output_sha256": _sha256(destination),
        "seconds": len(mixed) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "peak": round(float(np.max(np.abs(mixed))), 6),
        "samples_at_rail": int(np.sum(np.abs(mixed) >= 0.999999)),
        "phrases": len(phrases),
        "events": events,
        "rest_check": silent,
        "driven_by": ("the tune's continuous pitch curve, per hop; note pitches "
                      "are not used as a substitute"),
        "voice": "Hanakoloid CVVC bank by HanadaP - a guide voice, not the target",
    }


def _render_phrase(members: list, tune, bank, aliases, dll_path: Path,
                   breath: float, tension: float) -> tuple:
    """Synthesise one contiguous stretch. Its own engine instance, its own curve."""
    start = min(item.start_s for item, _ in members)
    end = max(item.end_s for item, _ in members)
    messages: list = []
    logger = _logger_type(lambda value: messages.append(
        value.decode("utf-8", errors="replace") if value else ""))

    dll = C.CDLL(str(dll_path))
    dll.PhraseSynthNew.argtypes = []
    dll.PhraseSynthNew.restype = C.c_void_p
    dll.PhraseSynthDelete.argtypes = [C.c_void_p]
    dll.PhraseSynthDelete.restype = None
    dll.PhraseSynthAddRequest.argtypes = [C.c_void_p, C.POINTER(Request), C.c_double,
                                          C.c_double, C.c_double, C.c_double,
                                          C.c_double, _logger_type]
    dll.PhraseSynthAddRequest.restype = None
    dll.PhraseSynthSetCurves.argtypes = [C.c_void_p, _pd, _pd, _pd, _pd, _pd,
                                         C.c_int, _logger_type]
    dll.PhraseSynthSetCurves.restype = None
    dll.PhraseSynthSynth.argtypes = [C.c_void_p, C.POINTER(_pf), _logger_type]
    dll.PhraseSynthSynth.restype = C.c_int

    synth = dll.PhraseSynthNew()
    if not synth:
        raise CarrierError("The engine would not start.")
    events: list = []
    try:
        for item, (head, tail) in members:
            local = item.start_s - start
            duration = item.duration_s
            tail_seconds = min(0.13, duration * 0.3) if tail else 0.0
            events.append(_add(dll, synth, logger, bank, aliases, head,
                               local, duration - tail_seconds, item.midi, False))
            if tail:
                events.append(_add(dll, synth, logger, bank, aliases, tail,
                                   local + duration - tail_seconds, tail_seconds,
                                   item.midi, True))

        count = int(math.ceil((end - start) / tune.hop_s)) + 1
        f0 = _curve_slice(tune, start, count)
        gender = np.full(count, 0.5)
        tension_curve = np.full(count, tension)
        breath_curve = np.full(count, breath)
        voicing = np.ones(count)
        dll.PhraseSynthSetCurves(
            synth, *[a.ctypes.data_as(_pd) for a in
                     (f0, gender, tension_curve, breath_curve, voicing)],
            count, logger)
        native = _pf()
        length = dll.PhraseSynthSynth(synth, C.byref(native), logger)
        if not 0 < length:
            raise CarrierError(f"The engine returned {length} samples: {messages[-3:]}")
        audio = np.ctypeslib.as_array(native, shape=(length,)).copy().astype(np.float64)
    finally:
        dll.PhraseSynthDelete(synth)

    if not np.isfinite(audio).all():
        raise CarrierError("The engine returned non-finite samples.")
    wanted = int(round((end - start) * SAMPLE_RATE))
    audio = audio[:wanted] if len(audio) >= wanted else np.pad(audio, (0, wanted - len(audio)))
    return audio, start, events


def _curve_slice(tune, start: float, count: int) -> np.ndarray:
    """The tune's own curve over this phrase.

    Unvoiced hops inside a phrase hold the last sung value so the engine has a
    defined pitch to resynthesise a consonant against; they are inside the
    phrase by construction, never across a rest, because phrases are split at
    rests before this is called.
    """
    begin = int(round(start / tune.hop_s))
    values = np.array(tune.f0_hz[begin:begin + count], dtype=float)
    voiced = np.array(tune.voiced[begin:begin + count], dtype=bool)
    if len(values) < count:
        pad = count - len(values)
        values = np.pad(values, (0, pad), mode="edge" if len(values) else "constant")
        voiced = np.pad(voiced, (0, pad), constant_values=False)
    filled = values.copy()
    last = 0.0
    for index in range(len(filled)):
        if voiced[index] and filled[index] > 0:
            last = filled[index]
        elif last > 0:
            filled[index] = last
    if not (filled > 0).any():
        raise CarrierError("This phrase has no voiced pitch to sing on.")
    first = float(filled[filled > 0][0])
    filled[filled <= 0] = first
    return np.ascontiguousarray(filled)


def _add(dll, synth, logger, bank, aliases, alias: str, position: float,
         duration: float, midi: float, coda: bool) -> dict:
    if alias not in aliases:
        raise CarrierError(f"The bank does not hold {alias!r}.")
    filename, (offset, consonant, cutoff, pre, overlap) = aliases[alias]
    raw = bank.read(filename)
    samples, rate = sf.read(io.BytesIO(raw), always_2d=True, dtype="float64")
    if rate != SAMPLE_RATE:
        raise CarrierError(f"{alias!r} is {rate} Hz; the bank must be {SAMPLE_RATE}.")
    samples = np.ascontiguousarray(samples[:, 0])
    if coda:
        finish = offset + consonant
        offset = max(offset, offset + pre - 45)
        window = finish - offset
        if window <= 30:
            raise CarrierError(f"{alias!r} has no usable coda window.")
        cutoff, consonant = -window, window
        velocity = 100 * (1 + math.log2(max(window, 1e-6) / max(duration * 1000, 1e-6)))
    else:
        velocity = 100
    region = -cutoff if cutoff < 0 else len(samples) / SAMPLE_RATE * 1000 - offset - cutoff
    if not 10 < region <= len(samples) / SAMPLE_RATE * 1000 - offset + 1:
        raise CarrierError(f"{alias!r} has an unusable region of {region:.1f} ms.")
    request = Request(
        sample_fs=SAMPLE_RATE, sample_length=len(samples),
        sample=samples.ctypes.data_as(_pd), tone=int(round(midi)), con_vel=velocity,
        offset=offset, required_length=duration * 1000 + 20, consonant=consonant,
        cut_off=cutoff, volume=100, modulation=0, tempo=100,
        flag_g=0, flag_O=0, flag_P=86, flag_Mt=0, flag_Mb=0, flag_Mv=100)
    fade = 20 if coda else 15
    dll.PhraseSynthAddRequest(synth, C.byref(request), position * 1000, 0,
                              duration * 1000, fade, fade, logger)
    return {"alias": alias, "sample": filename,
            "sample_sha256": hashlib.sha256(raw).hexdigest(),
            "position_s": round(position, 6), "duration_s": round(duration, 6),
            "coda": coda}


def _rest_levels(audio: np.ndarray, tune) -> dict:
    """Measure what actually landed in the rests.

    The reviewed implementation sang through them at RMS 0.311. This reports the
    number rather than asserting the property, so a regression is visible instead
    of being taken on trust.
    """
    voiced = np.array(tune.voiced, dtype=bool)
    if voiced.all() or not len(voiced):
        return {"rest_hops": 0, "rest_rms": 0.0, "rest_peak": 0.0,
                "note": "this tune has no rests"}
    per_hop = int(round(tune.hop_s * SAMPLE_RATE))
    mask = np.repeat(~voiced, per_hop)[:len(audio)]
    if len(mask) < len(audio):
        mask = np.pad(mask, (0, len(audio) - len(mask)), constant_values=False)
    inside = audio[mask]
    if not len(inside):
        return {"rest_hops": int((~voiced).sum()), "rest_rms": 0.0, "rest_peak": 0.0}
    return {"rest_hops": int((~voiced).sum()),
            "rest_rms": round(float(np.sqrt(np.mean(inside ** 2))), 6),
            "rest_peak": round(float(np.max(np.abs(inside))), 6),
            "note": "a rest that is not silent is a defect, not a style"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
