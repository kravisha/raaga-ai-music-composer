"""Stage 2 — shape the spoken Tamil onto the saved swara targets.

The score is the **morning packet's seven notes**
(`Raaga-Singing-Experiment-20260908/inputs/authored-tamil-phrase-proof.json`)
with the morning lower-register MIDI. It is **not** the separate 13-event
`nilavaai` fixture, and nothing here infers a tune from audio or matches pitch to
the accompanied SPB reference.

Four things this file does carefully, because a review found each of them wrong
in the first version:

1. **Contiguous coverage.** The spoken timeline is partitioned into syllable
   spans that touch end to end. No token is skipped — blanks included — so no
   spoken material is dropped at a syllable edge.
2. **Vowel boundaries are measured, not read off the spelling.** Tamil base
   consonants carry an inherent vowel, so `ல` has a nucleus even though nothing
   is written after it. The voiced region is found from Praat's pitch track and
   is labelled an **estimate** everywhere it appears.
3. **Exact durations by construction.** Timing is done in the sample domain:
   onset and coda samples are kept as they are and the nucleus is period-tiled
   to hit an exact sample count. A Praat DurationTier triangle was used before
   and does **not** integrate to the target — it averages toward it. Every
   syllable's realised length is measured and asserted.
5. **Pitch is preserved by construction.** Retiming uses phase-synchronous
   overlap-add with the output hop equal to the period; an earlier tiler shifted
   a 200 Hz tone to 266.67 Hz. `trials/periodic_retime.py` carries the invariant
   check.

4. **Failures fail.** If Praat cannot pitch a syllable, this stops. The previous
   version fell back to the unpitched speech and still called the result
   score-shaped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
OUT = WORKSPACE / "renders" / "tamil-tts-trial"
PSOLA_RUNTIME = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Tamil-CVVC-Trial-20260908"
                     r"\psola-runtime")
MORNING_PACKET = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Singing-Experiment-20260908"
                      r"\inputs\authored-tamil-phrase-proof.json")

MORNING_MIDI = [62, 51, 54, 55, 59, 56, 55]
MORNING_DURATIONS = [0.565, 0.2825, 0.2825, 0.565, 0.565, 1.13, 1.13]

#: Syllables as written, per word. Consonant characters only; the nucleus is
#: measured from the audio, because an inherent vowel is not written down.
SYLLABLES = [
    ("நிலவாய்", [["ந", "ி"], ["ல"], ["வ", "ா", "ய", "்"]]),
    ("நீ", [["ந", "ீ"]]),
    ("வந்தால்", [["வ", "ந", "்"], ["த", "ா", "ல", "்"]]),
    ("இரவும்", [["இ"], ["ர"], ["வ", "ு", "ம", "்"]]),
    ("பகல்", [["ப"], ["க", "ல", "்"]]),
    ("ஆகும்", [["ஆ"], ["க", "ு", "ம", "்"]]),
]

SR_TARGET = 44100
MIN_VOICED_S = 0.030
#: Below this, frame-to-frame band change is window-phase ripple rather than a
#: change in the sound. Measured: 4.3e-3 on a stationary harmonic, 0.034 at the
#: quietest inside real speech.
STATIONARY_FLOOR = 0.01


class ShapeError(Exception):
    """A refusal. The guide gate fails rather than degrading quietly."""


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def largest_remainder(total: int, weights: list) -> list:
    scale = sum(weights)
    exact = [total * w / scale for w in weights]
    share = [int(v) for v in exact]
    for i in sorted(range(len(weights)), key=lambda i: exact[i] - share[i],
                    reverse=True)[:total - sum(share)]:
        share[i] += 1
    for i, s in enumerate(share):
        if s == 0 and sum(share) < total:
            share[i] = 1
    return share


from trials.periodic_retime import retime as tile_to_length  # noqa: E402


def steady_region(np, piece, rate, times, freqs, origin, floor_s):
    """The part of a syllable that is actually a held vowel.

    "First-to-last voiced frame" was the wrong proxy. Tamil nasals, the
    approximants and the inherent vowel are all voiced, so that span is the
    whole syllable - it covered over 85% of the acoustic span in 11 of 13
    syllables here - and tiling it repeats the syllable instead of holding a
    vowel.

    A held vowel is steady in **two** ways at once: its pitch stops moving and
    its spectral shape stops moving. Consonants fail one or both - a nasal
    changes the spectrum sharply even while voicing continues, and a glide moves
    both. So this scores each frame on both and takes the longest run where both
    are quiet.

    Returns `(start_s, end_s, how)` in file time, or `(None, None, why)`.
    """
    window = int(0.025 * rate)
    hop = int(0.005 * rate)
    if len(piece) < window + hop:
        return None, None, "span shorter than one analysis window"

    count = (len(piece) - window) // hop + 1
    taper = np.hanning(window)
    bins = window // 2 + 1
    freqs_hz = np.fft.rfftfreq(window, 1 / rate)

    # Coarse log-spaced BANDS, not raw bins. A raw log spectrum shifts with the
    # window's phase against the waveform even when the sound is perfectly
    # stationary, so a steady harmonic tone reads as changing. Summing power
    # into bands removes that: a stationary harmonic signal now measures
    # stationary. Verified in `tests/test_steady_region.py`.
    edges = np.geomspace(80.0, max(rate / 2 - 1, 1000.0), 25)
    band_of = np.clip(np.searchsorted(edges, freqs_hz) - 1, 0, len(edges) - 2)
    bands = len(edges) - 1

    power = np.zeros((count, bands))
    for i in range(count):
        frame = piece[i * hop:i * hop + window] * taper
        magnitude = np.abs(np.fft.rfft(frame)) ** 2
        power[i] = np.bincount(band_of, weights=magnitude, minlength=bands)
    log = np.log(power + 1e-12)
    log -= log.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(log, axis=1, keepdims=True)
    shape = log / np.maximum(norm, 1e-9)
    spectral_change = np.ones(count)
    spectral_change[1:] = 1.0 - np.sum(shape[1:] * shape[:-1], axis=1)

    # Pitch movement over the same frames.
    frame_times = origin + (np.arange(count) * hop + window / 2) / rate
    f0 = np.interp(frame_times, times, np.where(freqs > 0, freqs, np.nan))
    pitch_change = np.ones(count)
    with np.errstate(invalid="ignore"):
        octaves = np.abs(np.diff(np.log2(np.where(f0 > 0, f0, np.nan))))
    pitch_change[1:] = np.nan_to_num(octaves, nan=1.0)
    voiced = np.nan_to_num(f0, nan=0.0) > 0

    # The threshold has to be relative to this syllable. An earlier version
    # capped it at an absolute 0.02, but the QUIETEST frame-to-frame change
    # inside a genuinely steady vowel measures about 0.034 here - the ceiling
    # sat below the natural floor, so nothing ever qualified and every syllable
    # silently fell back.
    if count > 3:
        moving = spectral_change[1:]
        quietest = float(np.nanmin(moving))
        upper = float(np.nanpercentile(moving, 75))
        # Anchored to the quietest change actually observed here, not to a
        # percentile and not to an absolute constant. A percentile collapses
        # when most changes are near zero - it then admits about half the frames
        # at random and a stationary signal reads as unsteady. An absolute cap
        # fails the other way: the quietest change inside a real vowel measures
        # about 0.034, well above the 0.02 ceiling an earlier version used, so
        # nothing ever qualified. The 1e-6 floor lets numerical noise through.
        # Two regimes, and the floor is what separates them. Band energies
        # still ripple slightly with the window's phase against the waveform -
        # measured at up to 4.3e-3 on a stationary harmonic tone, cycling every
        # four hops - while the quietest change inside real speech measures
        # about 0.034. A floor of 0.01 sits between the two: below it nothing is
        # meaningfully changing, and above it the anchored value governs.
        spectral_limit = max(STATIONARY_FLOOR,
                             quietest + 0.35 * (upper - quietest))
    else:
        spectral_limit = float(np.nanmax(spectral_change)) + 1.0
    steady = voiced & (spectral_change <= spectral_limit) & (pitch_change <= 0.03)

    # Energy per frame. Steadiness alone does not separate a vowel from a
    # sustained nasal - a held /m/ is steady too - but a vowel is the loud part
    # of its syllable and a nasal is not, so runs are scored by energy as well
    # as length.
    energy = np.array([float(np.sqrt(np.mean(
        piece[i * hop:i * hop + window] ** 2))) for i in range(count)])
    loudest = float(np.max(energy)) if count else 0.0

    runs, run_start = [], None
    for i, flag in enumerate(list(steady) + [False]):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            runs.append((run_start, i))
            run_start = None
    runs = [r for r in runs if r[1] - r[0] >= 2]
    if not runs:
        return None, None, "no steady run of two frames"

    def score(run):
        mean_energy = float(np.mean(energy[run[0]:run[1]]))
        return (run[1] - run[0]) * (mean_energy / max(loudest, 1e-12)) ** 2

    best_run = max(runs, key=score)
    mean_energy = float(np.mean(energy[best_run[0]:best_run[1]]))
    relative = mean_energy / max(loudest, 1e-12)
    if relative < 0.35:
        return None, None, (f"steadiest run is quiet ({relative:.2f} of the "
                            f"syllable peak) - likely a nasal, not the vowel")

    # best_run[1] is exclusive, so the last INCLUDED frame is best_run[1] - 1;
    # using the exclusive index put the end a whole hop past the audio.
    start = origin + (best_run[0] * hop) / rate
    last_frame = best_run[1] - 1
    end = origin + (last_frame * hop + window) / rate
    limit = origin + len(piece) / rate
    end = min(end, limit)
    start = min(start, end)
    if end - start < floor_s:
        return None, None, f"steadiest run only {end - start:.4f}s"
    return float(start), float(end), (
        f"ESTIMATE - steadiest loud run: band-spectrum change <= "
        f"{spectral_limit:.4f}, pitch change <= 3% per 5 ms, energy "
        f"{relative:.2f} of the syllable peak, {best_run[1] - best_run[0]} "
        f"frames. A sustained nasal can also be steady and loud, so this is not "
        f"a verified vowel.")


def main() -> int:
    sys.path.insert(0, str(PSOLA_RUNTIME))
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import numpy as np
    import parselmouth
    import soundfile as sf
    from scipy import signal as scipy_signal

    status = OUT / "shape-status.json"
    report = {"stage": "shape", "pid": os.getpid(),
              "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "score_source": str(MORNING_PACKET),
              "score_note": ("Morning packet's seven notes, morning lower-register "
                             "MIDI. NOT the 13-event nilavaai fixture; not inferred "
                             "from audio."),
              "psola": f"parselmouth {parselmouth.__version__} from psola-runtime",
              "training": False}

    def update(**fields):
        report.update(fields)
        status.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(json.dumps(fields, default=str, ensure_ascii=False), flush=True)

    update()
    packet = json.loads(MORNING_PACKET.read_text(encoding="utf-8"))
    timing = json.loads((OUT / "02-token-timing.json").read_text(encoding="utf-8"))
    speech, rate = sf.read(str(OUT / "01-tts-raw.wav"), dtype="float64")
    tokens = timing["tokens"]

    # -- contiguous partition: every token belongs to exactly one syllable ----
    letters = [t for t in tokens if t["token"] not in ("3",)]
    cursor, spans = 0, []
    for word, syllables in SYLLABLES:
        while cursor < len(letters) and letters[cursor]["token"] == " ":
            cursor += 1
        for chars in syllables:
            first = cursor
            for ch in chars:
                if cursor < len(letters) and letters[cursor]["token"] == ch:
                    cursor += 1
                else:
                    raise ShapeError(
                        f"token stream diverged at {ch!r} in {word!r}; refusing "
                        f"to guess a syllable boundary")
            spans.append({"word": word, "chars": "".join(chars),
                          "start_s": letters[first]["start_s"],
                          "end_s": letters[cursor - 1]["end_s"]})
    if len(spans) != 13:
        raise ShapeError(f"expected 13 syllables, partitioned {len(spans)}")

    # Close the gaps: each span runs to the next one's start, and the last to
    # the end of the audio, so nothing spoken is left out.
    for i, span in enumerate(spans):
        span["start_s"] = spans[i - 1]["end_s"] if i else 0.0
        span["end_s"] = (spans[i + 1]["start_s"] if i + 1 < len(spans)
                         else len(speech) / rate)
    for i, span in enumerate(spans[:-1]):
        span["end_s"] = spans[i + 1]["start_s"]
    spans[-1]["end_s"] = len(speech) / rate
    covered = sum(s["end_s"] - s["start_s"] for s in spans)
    if abs(covered - len(speech) / rate) > 1e-6:
        raise ShapeError(f"partition covers {covered:.6f}s of {len(speech)/rate:.6f}s")

    # -- trim the absorbed silence -------------------------------------------
    # Closing the gaps gave contiguous coverage, which is right: nothing spoken
    # is dropped. But it also pulled the silence BETWEEN words into whichever
    # syllable preceded it, and the next step would then protect that silence as
    # a coda. `வாய்` kept 0.2650 s of mostly inter-word gap as consonant and left
    # its vowel 17 ms, which is what produced a repeated syllable rather than a
    # held note. So the span is kept for coverage, and an ACOUSTIC sub-span is
    # measured inside it; only the acoustic part is treated as speech.
    frame = max(1, int(0.005 * rate))
    energy_floor = float(np.sqrt(np.mean(speech ** 2))) * 0.06
    for span in spans:
        begin, end = int(span["start_s"] * rate), int(span["end_s"] * rate)
        piece = speech[begin:end]
        if len(piece) < frame * 2:
            span["acoustic_start_s"], span["acoustic_end_s"] = span["start_s"], span["end_s"]
            span["gap_before_s"] = span["gap_after_s"] = 0.0
            continue
        count = len(piece) // frame
        levels = np.sqrt(np.mean(
            piece[:count * frame].reshape(count, frame) ** 2, axis=1))
        loud = np.flatnonzero(levels > energy_floor)
        if len(loud) == 0:
            span["acoustic_start_s"], span["acoustic_end_s"] = span["start_s"], span["end_s"]
            span["gap_before_s"] = span["gap_after_s"] = 0.0
            continue
        first = span["start_s"] + max(0, loud[0] - 1) * frame / rate
        last = span["start_s"] + min(count, loud[-1] + 2) * frame / rate
        span["acoustic_start_s"] = round(float(first), 6)
        span["acoustic_end_s"] = round(float(min(last, span["end_s"])), 6)
        span["gap_before_s"] = round(span["acoustic_start_s"] - span["start_s"], 6)
        span["gap_after_s"] = round(span["end_s"] - span["acoustic_end_s"], 6)
        span["acoustic_seconds"] = round(
            span["acoustic_end_s"] - span["acoustic_start_s"], 6)
        span["trim_note"] = ("silence at the span edges is recorded as gap, not "
                             "protected as consonant")

    # -- measured voiced region per syllable, inherent vowels included -------
    whole = parselmouth.Sound(speech, sampling_frequency=rate)
    pitch = whole.to_pitch(time_step=0.005, pitch_floor=60.0, pitch_ceiling=500.0)
    times = pitch.xs()
    values = pitch.selected_array["frequency"]
    for span in spans:
        begin = int(span["acoustic_start_s"] * rate)
        finish = int(span["acoustic_end_s"] * rate)
        low, high, how = steady_region(np, speech[begin:finish], rate, times,
                                       values, span["acoustic_start_s"],
                                       MIN_VOICED_S)
        inside = ((times >= span["acoustic_start_s"])
                  & (times < span["acoustic_end_s"]))
        voiced = times[inside][values[inside] > 0]
        if low is not None:
            span["nucleus_s"] = [low, high]
            span["nucleus_estimated_by"] = "steady region: " + how
        elif len(voiced) and voiced[-1] - voiced[0] >= MIN_VOICED_S:
            span["nucleus_s"] = [float(voiced[0]), float(voiced[-1])]
            span["nucleus_estimated_by"] = (
                "FALLBACK to voiced frames - " + how + " - this span may repeat "
                "consonant material when stretched")
        else:
            middle = (span["acoustic_start_s"] + span["acoustic_end_s"]) / 2
            half = max((span["acoustic_end_s"] - span["acoustic_start_s"]) * 0.25,
                       MIN_VOICED_S / 2)
            span["nucleus_s"] = [middle - half, middle + half]
            span["nucleus_estimated_by"] = "fallback: centre of the span, no voiced frames found"
        span["nucleus_is_an_estimate"] = True

    # -- 13 syllables onto 7 notes -------------------------------------------
    per_note = largest_remainder(len(spans), MORNING_DURATIONS)
    mapping, index, at = [], 0, 0.0
    for note_i, (midi, duration, take) in enumerate(
            zip(MORNING_MIDI, MORNING_DURATIONS, per_note)):
        members = spans[index:index + take]
        index += take
        weights = [max(m.get("acoustic_seconds",
                              m["end_s"] - m["start_s"]), 1e-3) for m in members]
        offset = at
        for member, weight in zip(members, weights):
            share = duration * weight / sum(weights)
            mapping.append(dict(member, note_index=note_i, midi=midi,
                                note_start_s=round(at, 6), note_duration_s=duration,
                                target_start_s=round(offset, 6),
                                target_seconds=round(share, 6),
                                spoken_seconds=round(
                                    member.get("acoustic_seconds",
                                               member["end_s"] - member["start_s"]), 6)))
            offset += share
        at += duration

    total = sum(MORNING_DURATIONS)
    (OUT / "04-score-mapping.json").write_text(json.dumps({
        "score_midi": MORNING_MIDI, "score_durations": MORNING_DURATIONS,
        "score_total_seconds": total, "packet": str(MORNING_PACKET),
        "packet_words": packet.get("draft_text"),
        "packet_note": ("The packet's seven slots and durations are reused; its "
                        "words are the morning phrase, not today's lyric."),
        "syllables_per_note": per_note,
        "mapping_rule": ("13 syllables over 7 notes. Each note takes a share by "
                         "largest remainder on its duration, then splits that "
                         "duration among its syllables in proportion to how long "
                         "each was actually spoken."),
        "mapping": mapping}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update(status="mapped", syllables=len(spans), notes=len(MORNING_MIDI),
           syllables_per_note=per_note, score_total_seconds=round(total, 6),
           spoken_total_seconds=round(len(speech) / rate, 6),
           coverage_seconds=round(covered, 6))

    # -- build ---------------------------------------------------------------
    out = np.zeros(int(round(total * rate)), dtype=np.float64)
    events, failures = [], []
    for item in mapping:
        begin = int(item.get("acoustic_start_s", item["start_s"]) * rate)
        end = int(item.get("acoustic_end_s", item["end_s"]) * rate)
        piece = speech[begin:end]
        target_samples = int(round(item["target_seconds"] * rate))
        hz = 440.0 * 2 ** ((item["midi"] - 69) / 12.0)
        if len(piece) < 128 or target_samples < 128:
            failures.append({"syllable": item["chars"], "why": "span too short"})
            continue

        # pitch only; duration is done in the sample domain.
        # Praat refuses a floor under 62.5 Hz and needs roughly three periods of
        # it, so short syllables are padded for the analysis and trimmed after.
        # The floor sits well under the lowest note here (MIDI 51 = 146.8 Hz).
        floor = 75.0
        need = int(rate * 3.5 / floor)
        pad = max(0, (need - len(piece)) // 2 + 1) if len(piece) < need else 0
        try:
            padded = (np.concatenate([np.zeros(pad), piece, np.zeros(pad)])
                      if pad else piece)
            sound = parselmouth.Sound(padded, sampling_frequency=rate)
            manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01,
                                                  floor, 500)
            tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
            parselmouth.praat.call(tier, "Remove points between", 0.0, sound.duration)
            parselmouth.praat.call(tier, "Add point", 0.0, hz)
            parselmouth.praat.call(tier, "Add point", sound.duration, hz)
            parselmouth.praat.call([tier, manipulation], "Replace pitch tier")
            pitched = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
            wave = np.asarray(pitched.values).reshape(-1)
            if pad:
                wave = wave[pad:pad + len(piece)] if len(wave) >= pad + len(piece)                     else wave[pad:]
        except Exception as exc:                      # noqa: BLE001
            failures.append({"syllable": item["chars"], "psola_error": repr(exc)})
            continue
        if not np.isfinite(wave).all() or float(np.max(np.abs(wave))) <= 0:
            failures.append({"syllable": item["chars"], "why": "resynthesis empty"})
            continue

        origin = item.get("acoustic_start_s", item["start_s"])
        n0 = int(max(0, (item["nucleus_s"][0] - origin) * rate))
        n1 = int(min(len(wave), (item["nucleus_s"][1] - origin) * rate))
        n0, n1 = min(n0, len(wave)), min(max(n1, n0 + 1), len(wave))
        onset, nucleus, coda = wave[:n0], wave[n0:n1], wave[n1:]
        period = max(32, int(rate / hz))
        fixed = len(onset) + len(coda)

        if target_samples > fixed + period:
            body = tile_to_length(nucleus, target_samples - fixed, period)
            built = np.concatenate([onset, body, coda])
            kept = "onset and coda kept at spoken length; nucleus retimed"
        else:
            # Not enough room to keep the consonants whole. Compress everything
            # and say so, rather than pretending they were protected.
            built = tile_to_length(wave, target_samples, period)
            kept = "target shorter than the consonants; whole syllable compressed"
        if len(built) != target_samples:
            raise ShapeError(f"{item['chars']!r} built {len(built)} of "
                             f"{target_samples} samples")

        at = int(round(item["target_start_s"] * rate))
        room = min(target_samples, len(out) - at)
        if room <= 0:
            raise ShapeError(f"{item['chars']!r} lands outside the score")
        fade = min(48, room // 6)
        window = np.ones(room)
        if fade > 1:
            window[:fade] = np.linspace(0, 1, fade)
            window[-fade:] = np.linspace(1, 0, fade)
        out[at:at + room] += built[:room] * window
        events.append({
            "syllable": item["chars"], "word": item["word"], "midi": item["midi"],
            "target_hz": round(hz, 3),
            "spoken_seconds": item["spoken_seconds"],
            "target_seconds": item["target_seconds"],
            "realised_seconds": round(room / rate, 6),
            "onset_samples_kept": int(len(onset)), "coda_samples_kept": int(len(coda)),
            "gap_trimmed_before_s": item.get("gap_before_s", 0.0),
            "gap_trimmed_after_s": item.get("gap_after_s", 0.0),
            "vowel_budget_s": round(item["target_seconds"]
                                    - (len(onset) + len(coda)) / rate, 6),
            "nucleus_estimated_by": item["nucleus_estimated_by"],
            "nucleus_is_an_estimate": True,
            "treatment": kept,
            "written_at_s": round(at / rate, 6)})

    if failures:
        update(status="failed", exit_code=1, failures=failures,
               gate="PSOLA or span failures - the guide is NOT score-shaped and "
                    "this stops before any conversion")
        raise ShapeError(f"{len(failures)} syllable(s) failed: {failures}")

    realised = sum(e["realised_seconds"] for e in events)
    peak = float(np.max(np.abs(out)))
    if peak <= 0:
        raise ShapeError("the shaped guide is silent")
    out *= 0.445 / peak
    resampled = scipy_signal.resample_poly(out, SR_TARGET, rate)
    guide = OUT / "13-shaped-guide-steady.wav"
    if guide.exists():
        raise ShapeError(f"{guide} exists; guides are not overwritten")
    sf.write(str(guide), resampled, SR_TARGET, subtype="PCM_24")

    (OUT / "06-articulation.json").write_text(json.dumps(
        {"events": events,
         "note": ("Nucleus boundaries are ESTIMATES from Praat's voiced-frame "
                  "endpoints, including inherent vowels that are not written. "
                  "Onset and coda keep their spoken sample counts where the "
                  "target allows. That is a timing decision, NOT acoustic "
                  "validation: nothing here proves a nasal or voiced coda "
                  "survived, and voiced-frame endpoints are not phoneme "
                  "boundaries."),
         "realised_total_seconds": round(realised, 6),
         "score_total_seconds": round(total, 6)},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    update(status="completed", exit_code=0,
           finish_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           realised_total_seconds=round(realised, 6),
           duration_error_seconds=round(abs(realised - total), 9),
           guide={"path": str(guide), "sha256": sha(guide),
                  "sample_rate": SR_TARGET,
                  "seconds": round(len(resampled) / SR_TARGET, 6),
                  "peak": round(float(np.max(np.abs(resampled))), 6),
                  "rms_dbfs": round(float(20 * np.log10(float(np.sqrt(np.mean(
                      resampled ** 2))))), 4)},
           events=events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
