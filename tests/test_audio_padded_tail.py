"""Artificial zero-padding does not require thousands of zero-input FFTs."""

import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx import audio


def full_padded_reference(samples, n_mels, padding):
    # Frozen numerical frontend from commit 76e3f46 (Apple's original MLX path).
    # Keep the full padded FFT and GEMM here rather than the optimized helper.
    with mx.stream(mx.cpu):
        samples = audio.prepare_audio(samples)
        if padding > 0:
            samples = mx.pad(samples, (0, padding))
        window = mx.array(np.hanning(audio.N_FFT + 1)[:-1])
        half = audio.N_FFT // 2
        samples = mx.concatenate(
            [samples[1 : half + 1][::-1], samples, samples[-(half + 1) : -1][::-1]]
        )
        frames = (samples.size - audio.N_FFT + audio.HOP_LENGTH) // audio.HOP_LENGTH
        strided = mx.as_strided(
            samples, shape=(frames, audio.N_FFT), strides=(audio.HOP_LENGTH, 1)
        )
        frequencies = mx.fft.rfft(strided * window)
        magnitudes = frequencies[:-1].abs().square()
        mel = magnitudes @ audio.mel_filters(n_mels).T
        result = mx.maximum(mel, 1e-10).log10()
        result = mx.maximum(result, result.max() - 8.0)
        return (result + 4.0) / 4.0


def test_fft_work_is_bounded_by_real_samples_and_centered_window_guard(monkeypatch):
    sizes = []
    original = audio.stft

    def observe(samples, *args, **kwargs):
        sizes.append(samples.size)
        return original(samples, *args, **kwargs)

    monkeypatch.setattr(audio, "stft", observe)
    samples = mx.array(np.random.default_rng(2).normal(size=32161).astype(np.float32))
    actual = audio.log_mel_spectrogram(samples, padding=audio.N_SAMPLES)
    mx.eval(actual)
    assert sizes and max(sizes) <= samples.size + audio.N_FFT


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize(
    "samples,padding",
    [
        (0, 480000),
        (1, 480000),
        (159, 480000),
        (201, 0),
        (400, 1),
        (400, 399),
        (401, 400),
        (401, 401),
        (31999, 480000),
        (32000, 480000),
        (32119, 480000),
        (32120, 480000),
        (32159, 480000),
        (32161, 480000),
        (480123, 480000),
    ],
)
def test_padded_frontend_exactly_matches_full_fft_reference(
    n_mels, samples, padding, dtype
):
    waveform = np.random.default_rng(7).normal(size=samples).astype(dtype)
    expected = full_padded_reference(waveform, n_mels, padding)
    actual = audio.log_mel_spectrogram(waveform, n_mels=n_mels, padding=padding)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    "kind", ["zero", "dc", "quiet", "first_impulse", "last_impulse"]
)
@pytest.mark.parametrize("samples", [1, 159, 32001, 32159])
def test_zero_dc_quiet_and_endpoint_energy_keep_exact_global_clamp(kind, samples):
    waveform = np.zeros(samples, np.float32)
    if kind == "dc":
        waveform.fill(0.25)
    elif kind == "quiet":
        waveform[:] = np.random.default_rng(3).normal(size=samples) * 1e-8
    elif kind == "first_impulse":
        waveform[0] = 1
    elif kind == "last_impulse":
        waveform[-1] = 1
    expected = full_padded_reference(waveform, 80, audio.N_SAMPLES)
    actual = audio.log_mel_spectrogram(waveform, padding=audio.N_SAMPLES)
    assert mx.array_equal(actual, expected).item()
