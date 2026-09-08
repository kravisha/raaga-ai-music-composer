"""The closing consonant belongs within its note, including very short notes."""
import numpy as np
import pytest

from raagacomposer.voice.renderer import _close_with_consonant


@pytest.mark.parametrize("coda", ["n", "s", "l", "t"])
@pytest.mark.parametrize("seconds", [0.01, 0.02, 0.10])
def test_closing_consonant_never_changes_audio_outside_its_note(coda, seconds):
    sr = 22050
    a = int(0.20 * sr)
    b = a + int(seconds * sr)
    original = np.full(sr, 0.12, dtype=np.float32)
    actual = _close_with_consonant(original.copy(), coda, a, b, sr,
                                   np.random.default_rng(2026), 0.8)
    assert np.array_equal(actual[:a], original[:a]), "Closing consonant changed earlier music"
    assert np.array_equal(actual[b:], original[b:]), "Closing consonant spilled past its note into following music"
    assert not np.array_equal(actual[a:b], original[a:b]), "No closing consonant was applied"
