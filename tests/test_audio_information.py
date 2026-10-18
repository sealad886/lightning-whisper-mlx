"""Exact acoustic-information proof excludes no nonconstant finite waveform."""

import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.audio import is_constant_audio, prepare_audio


@pytest.mark.parametrize("value", [0.0, 0.5, -0.75])
def test_constant_pcm_including_dc_is_proven_without_padding(value):
    samples = mx.full((16000,), value)
    assert is_constant_audio(samples, 0, 16000)


def test_nonconstant_very_quiet_pcm_is_not_empty():
    samples = mx.zeros(16000)
    samples[100] = 1e-12
    assert not is_constant_audio(samples, 0, 16000)


def test_gate_includes_real_stft_neighbors_but_not_the_whole_recording():
    samples = mx.zeros(32000)
    samples[15900] = 1e-12
    assert not is_constant_audio(samples, 16000, 32000)
    assert is_constant_audio(samples, 16300, 32000)


def test_preparation_reuses_mlx_storage_and_dtype():
    samples = mx.array([0.0, 1.0], dtype=mx.float16)
    prepared = prepare_audio(samples)
    assert prepared is samples
    assert prepared.dtype == mx.float16


@pytest.mark.parametrize("samples", [np.array([[0.0, 1.0]], np.float32), np.array(0.0)])
def test_nonmono_waveform_has_clear_validation(samples):
    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_audio(samples)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_rejected_and_never_proven_silent(value):
    samples = mx.array([value, value])
    assert not is_constant_audio(samples, 0, 2)
    with pytest.raises(ValueError, match="finite"):
        prepare_audio(samples)
