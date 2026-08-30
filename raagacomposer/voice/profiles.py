"""Voice profile manager (spec section 4 step 6).

Ships a small set of built-in voices and lets the creator add their own from
supplied recordings.  A user profile is derived by analysing the recording --
median pitch, brightness and noise floor -- and stored under the application
config directory, so changing singer never touches the tune.

Only recordings the creator supplies are used.  Nothing here downloads or
imitates a third-party voice.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.logging_setup import get_logger
from ..core.models import VoiceProfile
from ..core.serde import from_jsonable, to_jsonable
from ..core.settings import config_dir
from ..music.theory import freq_to_midi

log = get_logger("voice.profiles")

BUILTIN: List[VoiceProfile] = [
    VoiceProfile(id="voice_female_warm", name="Female - Warm", gender="female",
                 base_midi=62, range_low=55, range_high=81, formant_shift=1.16,
                 breathiness=0.13, vibrato_rate=5.4, vibrato_depth=0.17,
                 brightness=1.0,
                 notes="Rounded playback-style female voice; the default."),
    VoiceProfile(id="voice_female_bright", name="Female - Bright", gender="female",
                 base_midi=64, range_low=57, range_high=84, formant_shift=1.22,
                 breathiness=0.09, vibrato_rate=5.9, vibrato_depth=0.2,
                 brightness=1.25, notes="Forward and cutting; good for a chorus."),
    VoiceProfile(id="voice_male_warm", name="Male - Warm", gender="male",
                 base_midi=50, range_low=41, range_high=67, formant_shift=1.0,
                 breathiness=0.12, vibrato_rate=5.0, vibrato_depth=0.15,
                 brightness=0.95, notes="Chest-forward male voice."),
    VoiceProfile(id="voice_male_bright", name="Male - Bright", gender="male",
                 base_midi=52, range_low=43, range_high=71, formant_shift=1.05,
                 breathiness=0.08, vibrato_rate=5.5, vibrato_depth=0.18,
                 brightness=1.2, notes="Higher male register with more edge."),
    VoiceProfile(id="voice_light", name="Light / Youthful", gender="neutral",
                 base_midi=67, range_low=60, range_high=88, formant_shift=1.3,
                 breathiness=0.16, vibrato_rate=5.2, vibrato_depth=0.12,
                 brightness=1.1, notes="Light timbre for gentle or innocent songs."),
]


class VoiceProfileManager:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config_dir() / "voices.json"
        self._user: Dict[str, VoiceProfile] = {}
        self.load()

    # -- storage -----------------------------------------------------------
    def load(self) -> None:
        self._user.clear()
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("cannot read voice profiles: %s", exc)
            return
        for entry in data.get("voices", []):
            try:
                vp = from_jsonable(VoiceProfile, entry)
                vp.builtin = False
                self._user[vp.id] = vp
            except Exception as exc:  # noqa: BLE001
                log.error("bad voice profile %r: %s", entry.get("name"), exc)

    def save(self) -> None:
        payload = {"voices": [to_jsonable(v) for v in self._user.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- lookup ------------------------------------------------------------
    def all(self) -> List[VoiceProfile]:
        return list(BUILTIN) + list(self._user.values())

    def get(self, profile_id: str) -> Optional[VoiceProfile]:
        for v in self.all():
            if v.id == profile_id:
                return v
        return None

    def by_name(self, name: str) -> Optional[VoiceProfile]:
        """Find a voice by name, then by gender word.

        Matching is on whole words: "male" must not select "Female - Warm".
        """
        low = (name or "").strip().lower()
        if not low:
            return None
        for v in self.all():
            if low == v.name.lower():
                return v
        pattern = re.compile(rf"(?<![a-z]){re.escape(low)}(?![a-z])")
        for v in self.all():
            if pattern.search(v.name.lower()):
                return v
        for v in self.all():
            if pattern.search(v.gender.lower()):
                return v
        return None

    def default(self, gender_hint: str = "") -> VoiceProfile:
        hint = (gender_hint or "").strip().lower()
        if hint:
            pattern = re.compile(rf"(?<![a-z]){re.escape(hint)}(?![a-z])")
            for v in self.all():
                if pattern.search(v.gender.lower()):
                    return v
        return BUILTIN[0]

    # -- editing -----------------------------------------------------------
    def add(self, profile: VoiceProfile) -> VoiceProfile:
        profile.builtin = False
        self._user[profile.id] = profile
        self.save()
        return profile

    def update(self, profile: VoiceProfile) -> VoiceProfile:
        if profile.builtin:
            raise ValueError("Built-in voices cannot be edited; duplicate it first.")
        self._user[profile.id] = profile
        self.save()
        return profile

    def duplicate(self, profile_id: str, new_name: str) -> VoiceProfile:
        src = self.get(profile_id)
        if src is None:
            raise KeyError(profile_id)
        clone = from_jsonable(VoiceProfile, to_jsonable(src))
        clone.id = f"voice_{abs(hash(new_name)) % 10 ** 10:010d}"
        clone.name = new_name
        clone.builtin = False
        return self.add(clone)

    def remove(self, profile_id: str) -> bool:
        if profile_id in self._user:
            del self._user[profile_id]
            self.save()
            return True
        return False

    # -- analysis ----------------------------------------------------------
    def create_from_recording(self, wav_paths: List[str], name: str,
                              gender: str = "") -> VoiceProfile:
        """Derive a profile from creator-supplied recordings."""
        import soundfile as sf

        f0s: List[float] = []
        brightness: List[float] = []
        noise_floor: List[float] = []
        used: List[str] = []
        for p in wav_paths:
            try:
                audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            except Exception as exc:  # noqa: BLE001
                log.error("cannot read %s: %s", p, exc)
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            used.append(str(p))
            f0, bright, floor = _analyse(audio, sr)
            f0s.extend(f0)
            brightness.append(bright)
            noise_floor.append(floor)

        if not f0s:
            raise ValueError("No usable pitch found in the supplied recordings.")

        median_f0 = float(np.median(f0s))
        low_f0 = float(np.percentile(f0s, 5))
        high_f0 = float(np.percentile(f0s, 95))
        base = int(round(freq_to_midi(median_f0)))
        detected_gender = gender or ("male" if median_f0 < 165 else "female")
        shift = 1.0 if detected_gender == "male" else 1.16
        profile = VoiceProfile(
            id=f"voice_user_{abs(hash(name)) % 10 ** 8:08d}",
            name=name or "My Voice",
            gender=detected_gender,
            base_midi=base,
            range_low=max(30, int(round(freq_to_midi(low_f0))) - 3),
            range_high=min(96, int(round(freq_to_midi(high_f0))) + 5),
            formant_shift=shift,
            breathiness=float(np.clip(np.mean(noise_floor) * 4.0, 0.05, 0.35)),
            brightness=float(np.clip(np.mean(brightness) / 2200.0, 0.7, 1.4)),
            source_samples=used,
            builtin=False,
            notes=f"Derived from {len(used)} recording(s); median pitch "
                  f"{median_f0:.0f} Hz.")
        return self.add(profile)


def _analyse(audio: np.ndarray, sr: int) -> tuple:
    """Rough per-file pitch track, spectral centroid and noise floor."""
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    f0s: List[float] = []
    centroids: List[float] = []
    energies: List[float] = []
    for i in range(0, max(0, len(audio) - frame), hop):
        seg = audio[i:i + frame]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        energies.append(rms)
        if rms < 0.01:
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        lo = int(sr / 500)
        hi = min(len(corr) - 1, int(sr / 70))
        if hi <= lo:
            continue
        peak = int(np.argmax(corr[lo:hi])) + lo
        if corr[peak] > 0.3 * corr[0] and peak > 0:
            f0s.append(sr / peak)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        if spec.sum() > 0:
            centroids.append(float((spec * freqs).sum() / spec.sum()))
    floor = float(np.percentile(energies, 10)) if energies else 0.02
    return f0s, (float(np.mean(centroids)) if centroids else 1800.0), floor
