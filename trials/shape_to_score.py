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


def tile_to_length(np, block, target: int, period: int) -> "np.ndarray":
    """Stretch or compress a periodic block to exactly `target` samples.

    Whole periods are repeated or dropped with short crossfades. Because the
    nucleus has already been flattened to one known pitch, a period is a known
    integer, and the result lands on the exact sample count rather than near it.
    """
    if target <= 0:
        return np.zeros(0)
    if len(block) == 0:
        return np.zeros(target)
    period = max(32, min(period, max(64, len(block) // 2)))
    if len(block) < 2 * period:
        index = (np.arange(target) * (len(block) - 1) / max(target - 1, 1))
        return np.interp(index, np.arange(len(block)), block)
    out = np.zeros(target)
    fade = max(8, period // 4)
    written, read = 0, 0
    while written < target:
        take = min(period, target - written, len(block) - read)
        if take <= 0:
            read = 0
            continue
        segment = block[read:read + take].copy()
        if written and fade < take:
            segment[:fade] *= np.linspace(0, 1, fade)
            out[written:written + fade] *= np.linspace(1, 0, fade)
        out[written:written + take] += segment
        written += take - (fade if (written and fade < take) else 0)
        read += take
        if read >= len(block) - period:
            read = max(0, len(block) - 2 * period)
    return out[:target]


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

    # -- measured voiced region per syllable, inherent vowels included -------
    whole = parselmouth.Sound(speech, sampling_frequency=rate)
    pitch = whole.to_pitch(time_step=0.005, pitch_floor=60.0, pitch_ceiling=500.0)
    times = pitch.xs()
    values = pitch.selected_array["frequency"]
    for span in spans:
        inside = (times >= span["start_s"]) & (times < span["end_s"])
        voiced = times[inside][values[inside] > 0]
        if len(voiced) and voiced[-1] - voiced[0] >= MIN_VOICED_S:
            span["nucleus_s"] = [float(voiced[0]), float(voiced[-1])]
            span["nucleus_estimated_by"] = "Praat pitch track, voiced frames"
        else:
            middle = (span["start_s"] + span["end_s"]) / 2
            half = max((span["end_s"] - span["start_s"]) * 0.25, MIN_VOICED_S / 2)
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
        weights = [max(m["end_s"] - m["start_s"], 1e-3) for m in members]
        offset = at
        for member, weight in zip(members, weights):
            share = duration * weight / sum(weights)
            mapping.append(dict(member, note_index=note_i, midi=midi,
                                note_start_s=round(at, 6), note_duration_s=duration,
                                target_start_s=round(offset, 6),
                                target_seconds=round(share, 6),
                                spoken_seconds=round(member["end_s"] - member["start_s"], 6)))
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
        begin, end = int(item["start_s"] * rate), int(item["end_s"] * rate)
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

        n0 = int(max(0, (item["nucleus_s"][0] - item["start_s"]) * rate))
        n1 = int(min(len(wave), (item["nucleus_s"][1] - item["start_s"]) * rate))
        n0, n1 = min(n0, len(wave)), min(max(n1, n0 + 1), len(wave))
        onset, nucleus, coda = wave[:n0], wave[n0:n1], wave[n1:]
        period = max(32, int(rate / hz))
        fixed = len(onset) + len(coda)

        if target_samples > fixed + period:
            body = tile_to_length(np, nucleus, target_samples - fixed, period)
            built = np.concatenate([onset, body, coda])
            kept = "onset and coda kept at spoken length; nucleus retimed"
        else:
            # Not enough room to keep the consonants whole. Compress everything
            # and say so, rather than pretending they were protected.
            built = tile_to_length(np, wave, target_samples, period)
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
    guide = OUT / "05-shaped-guide.wav"
    if guide.exists():
        guide = OUT / "05-shaped-guide-v2.wav"
    sf.write(str(guide), resampled, SR_TARGET, subtype="PCM_24")

    (OUT / "06-articulation.json").write_text(json.dumps(
        {"events": events,
         "note": ("Nucleus boundaries are ESTIMATES from Praat's pitch track, "
                  "including inherent vowels that are not written. Onset and coda "
                  "keep their spoken sample counts wherever the target allows."),
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
