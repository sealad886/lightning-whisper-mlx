"""Public transcription contracts, using deterministic model/audio substitutes."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

tr = importlib.import_module("lightning_whisper_mlx.transcribe")


class Tokenizer:
    eot = 1000
    timestamp_begin = 10000

    def encode(self, text):
        return [99]

    def decode(self, tokens):
        return "".join(f"{token} " for token in tokens if token < self.eot)


@pytest.fixture
def transcription(monkeypatch):
    class Model:
        dims = SimpleNamespace(n_mels=1, n_audio_ctx=1500)
        is_multilingual = False
        num_languages = 0

        def __init__(self):
            self.calls = []
            self.encodings = 0
            self.answer = lambda start, options: {}

        def decode(self, batch, options, audio_features=None):
            if audio_features is None:
                self.encodings += 1
            starts = [int(batch[i, 0, 0].item()) for i in range(batch.shape[0])]
            self.calls.append((starts, options))
            return [
                SimpleNamespace(
                    **{
                        "tokens": [start // 3000 + 1],
                        "temperature": options.temperature,
                        "avg_logprob": -0.1,
                        "no_speech_prob": 0.0,
                        "compression_ratio": 1.0,
                        "audio_features": batch[i],
                        **self.answer(start, options),
                    }
                )
                for i, start in enumerate(starts)
            ]

    model = Model()
    monkeypatch.setattr(tr.ModelHolder, "get_model", lambda *args: model)
    monkeypatch.setattr(tr, "get_tokenizer", lambda *args, **kwargs: Tokenizer())

    def run(seconds=90, waveform=None, **kwargs):
        mel = mx.array(
            np.arange(round(seconds * 100) + 3000, dtype=np.float32)[:, None]
        )
        monkeypatch.setattr(
            tr, "log_mel_spectrogram_from_prepared", lambda *args, **kwargs: mel
        )
        if waveform is None:
            waveform = np.resize(
                np.array([0.0, 1.0], np.float32), round(seconds * 16000)
            )
        return tr.transcribe_audio(
            waveform, language=kwargs.pop("language", "en"), fp16=False, **kwargs
        )

    return model, run


@pytest.mark.parametrize("seconds", [30, 90, 65])
@pytest.mark.parametrize("batch_size", [1, 2, 12])
def test_windows_preserve_all_text_and_per_item_seconds(
    transcription, seconds, batch_size
):
    model, run = transcription
    result = run(seconds, batch_size=batch_size, condition_on_previous_text=False)
    expected_starts = list(range(0, round(seconds * 100), 3000))
    assert [start for starts, _ in model.calls for start in starts] == expected_starts
    assert [s["seek"] for s in result["segments"]] == expected_starts
    assert [s["start"] for s in result["segments"]] == [
        s / 100 for s in expected_starts
    ]
    assert [s["end"] for s in result["segments"]] == [
        min(s / 100 + 30, seconds) for s in expected_starts
    ]
    assert result["text"] == "".join(f"{i + 1} " for i in range(len(expected_starts)))
    assert all(s["tokens"] and s["avg_logprob"] == -0.1 for s in result["segments"])


def test_all_clips_honor_nonzero_starts_and_exclusive_ends(transcription):
    model, run = transcription
    result = run(100, clip_timestamps=[30, 45, 60, 75], batch_size=12)
    assert [starts for starts, _ in model.calls] == [[3000], [6000]]
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(30, 45), (60, 75)]


@pytest.mark.parametrize(
    "tokens, expected_text, expected_intervals",
    [
        ([10000, 1, 11500], "1 ", [(30, 40)]),
        ([10000, 1, 11200, 11200, 2, 11500], "1 ", [(30, 40)]),
        ([10000, 1, 10200], "1 ", [(30, 34)]),
    ],
)
def test_timestamp_tokens_cannot_extend_selected_clip(
    transcription, tokens, expected_text, expected_intervals
):
    model, run = transcription
    model.answer = lambda *args: {"tokens": tokens}
    result = run(45, clip_timestamps=[30, 40])
    assert result["text"] == expected_text
    assert [(s["start"], s["end"]) for s in result["segments"]] == expected_intervals
    assert len(model.calls) == 1


def test_word_alignment_is_bounded_before_updating_later_clip_state(
    transcription, monkeypatch
):
    model, run = transcription
    model.answer = lambda *args: {"tokens": [10000, 1, 11500]}
    alignment_state = []

    def align(*, segments, last_speech_timestamp, **kwargs):
        alignment_state.append(last_speech_timestamp)
        segment = segments[0]
        start = segment["seek"] / 100
        # Alignment/segment heuristics can shift both sides of the window.
        segment.update(start=start - 1, end=start + 12)
        segment["words"] = [
            {"word": "hello", "start": start - 1, "end": start + 12, "probability": 0.9}
        ]

    monkeypatch.setattr(tr, "add_word_timestamps", align)
    result = run(75, clip_timestamps=[30, 40, 60, 70], word_timestamps=True)
    assert alignment_state == [0.0, 40.0]
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(30, 40), (60, 70)]
    assert [(w["start"], w["end"]) for s in result["segments"] for w in s["words"]] == [
        (30, 40),
        (60, 70),
    ]
    assert result["text"] == "1 1 "


def test_conditioned_windows_use_committed_previous_text(transcription):
    model, run = transcription
    run(65, batch_size=12, initial_prompt="names")
    assert [starts for starts, _ in model.calls] == [[0], [3000], [6000]]
    assert [options.prompt for _, options in model.calls] == [[99], [99, 1], [99, 1, 2]]


def test_initial_prompt_applies_only_to_first_independent_window(transcription):
    model, run = transcription
    run(65, batch_size=12, condition_on_previous_text=False, initial_prompt="names")
    assert [starts for starts, _ in model.calls] == [[0], [3000, 6000]]
    assert [options.prompt for _, options in model.calls] == [[99], []]


def test_unfinished_timestamp_tail_resumes_before_later_batch_windows(transcription):
    model, run = transcription
    model.answer = lambda start, options: (
        {"tokens": [10000, 1, 11000, 11000, 9]} if start == 0 else {"tokens": [2]}
    )
    result = run(65, batch_size=12, condition_on_previous_text=False)
    assert [starts for starts, _ in model.calls] == [[0, 3000, 6000], [2000, 5000]]
    assert [s["seek"] for s in result["segments"]] == [0, 2000, 5000]
    assert result["text"] == "1 2 2 "


def test_fallback_honors_schedule_and_retries_only_failed_rows(transcription):
    model, run = transcription
    model.answer = lambda start, options: {
        "compression_ratio": 3.0 if start == 0 and options.temperature < 0.6 else 1.0
    }
    run(
        60,
        batch_size=2,
        condition_on_previous_text=False,
        temperature=(0, 0.3, 0.6),
        best_of=2,
    )
    assert [(starts, options.temperature) for starts, options in model.calls] == [
        ([0, 3000], 0),
        ([0], 0.3),
        ([0], 0.6),
    ]
    assert [options.best_of for _, options in model.calls] == [None, 2, 2]
    assert model.encodings == 1


def test_independent_recordings_batch_without_sharing_conditioned_prompts(
    transcription, monkeypatch
):
    model, _ = transcription
    monkeypatch.setattr(
        tr,
        "log_mel_spectrogram_from_prepared",
        lambda audio, **kwargs: mx.array(
            (np.arange(9000, dtype=np.float32) + audio[0].item() * 15000)[:, None]
        ),
    )
    result = tr.transcribe_many_audio(
        [
            np.resize(np.array([0.0, 1.0], np.float32), 960000),
            np.resize(np.array([2.0, 3.0], np.float32), 960000),
        ],
        model_instance=model,
        language="en",
        fp16=False,
        batch_size=2,
        temperature=0.0,
    )
    # First windows share a compatible empty prompt. Subsequent prompts differ,
    # so each recording continues independently without changing its semantics.
    assert [starts for starts, _ in model.calls] == [[0, 30000], [3000], [33000]]
    assert [options.prompt for _, options in model.calls] == [[], [1], [11]]
    assert [row["text"] for row in result] == ["1 2 ", "11 12 "]
    assert [[s["seek"] for s in row["segments"]] for row in result] == [[0, 3000]] * 2


def test_empty_request_batch_does_not_load_a_model(monkeypatch):
    monkeypatch.setattr(tr.ModelHolder, "get_model", lambda *a: pytest.fail("loaded"))
    assert tr.transcribe_many_audio([]) == []


def test_scalar_temperature_has_no_implicit_retry(transcription):
    model, run = transcription
    model.answer = lambda *args: {"compression_ratio": 3.0}
    run(10, temperature=0.3, best_of=2)
    assert [o.temperature for _, o in model.calls] == [0.3]


def test_confident_speech_with_high_no_speech_prob_still_retries(transcription):
    model, run = transcription
    model.answer = lambda start, options: {
        "no_speech_prob": 0.9,
        "compression_ratio": 3.0 if options.temperature == 0 else 1.0,
    }
    assert run(10)["text"] == "1 "
    assert [o.temperature for _, o in model.calls] == [0.0, 1.0]


def test_word_timestamps_use_existing_alignment_and_absolute_clip_offset(
    transcription, monkeypatch
):
    from lightning_whisper_mlx import timing

    model, run = transcription
    model.answer = lambda *args: {"tokens": [10000, 1, 10500]}
    aligned = []

    def align(model, tokenizer, tokens, mel, num_frames, **kwargs):
        aligned.append((tokens, int(mel[0, 0].item()), num_frames))
        assert mx.array_equal(kwargs["audio_features"], mel).item()
        return [timing.WordTiming("hello", [1], 1.0, 2.0, 0.99)]

    monkeypatch.setattr(timing, "find_alignment", align)
    result = run(45, clip_timestamps=[30, 40], word_timestamps=True)
    assert aligned == [([1], 3000, 1000)]
    word = result["segments"][0]["words"][0]
    assert word == {"word": "hello", "start": 31.0, "end": 32.0, "probability": 0.99}


def test_hallucination_silence_resumes_after_leading_gap(transcription, monkeypatch):
    from lightning_whisper_mlx import timing

    model, run = transcription
    model.answer = lambda *args: {"tokens": [10000, 1, 11000]}

    def align(model, tokenizer, tokens, mel, num_frames, **kwargs):
        if int(mel[0, 0].item()) == 0:
            return [timing.WordTiming("noise", [1], 10.0, 10.05, 0.01)]
        return [timing.WordTiming("speech", [1], 1.0, 2.0, 0.99)]

    monkeypatch.setattr(timing, "find_alignment", align)
    result = run(20, word_timestamps=True, hallucination_silence_threshold=2)
    assert [starts for starts, _ in model.calls] == [[0], [1000]]
    assert [w["word"] for s in result["segments"] for w in s["words"]] == ["speech"]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_batch_size_fails_before_loading(transcription, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        transcription[1](batch_size=batch_size)


def test_model_cache_reloads_on_dtype_or_artifact_change(monkeypatch, tmp_path):
    for filename in ("config.json", "weights.npz"):
        (tmp_path / filename).write_bytes(b"initial")
    calls = []
    monkeypatch.setattr(
        tr, "load_model", lambda path, dtype: calls.append(dtype) or object()
    )
    monkeypatch.setattr(tr.ModelHolder, "model", None)
    monkeypatch.setattr(tr.ModelHolder, "model_identity", None)
    first = tr.ModelHolder.get_model(str(tmp_path), mx.float16)
    assert tr.ModelHolder.get_model(str(tmp_path / "."), mx.float16) is first
    assert tr.ModelHolder.get_model(str(tmp_path), mx.float32) is not first
    (tmp_path / "weights.npz").write_bytes(b"replacement-weights")
    tr.ModelHolder.get_model(str(tmp_path), mx.float32)
    assert calls == [mx.float16, mx.float32, mx.float32]


@pytest.mark.parametrize("seconds", [30, 65])
def test_auto_language_reuses_first_full_window_in_batch(transcription, seconds):
    model, run = transcription
    model.is_multilingual, model.num_languages = True, 99
    encoded = []

    def encode(batch):
        encoded.extend(batch[:, 0, 0].tolist())
        return batch

    model.embed_audio = encode
    model.detect_language = lambda features: (None, {"en": 1.0})
    run(seconds, language=None, batch_size=3, condition_on_previous_text=False)
    assert encoded == list(range(0, round(seconds * 100), 3000))
    assert model.encodings == 0


def test_auto_language_features_do_not_replace_a_clipped_partial_window(transcription):
    model, run = transcription
    model.is_multilingual, model.num_languages = True, 99
    model.embed_audio = lambda batch: batch
    model.detect_language = lambda features: (None, {"en": 1.0})
    run(60, language=None, batch_size=3, clip_timestamps=[0, 10])
    assert model.encodings == 1


@pytest.mark.parametrize("seconds", [0.01, 10, 29.99])
def test_short_auto_language_does_not_reuse_differently_padded_features(
    transcription, seconds
):
    model, run = transcription
    model.is_multilingual, model.num_languages = True, 99
    language_inputs, transcription_inputs, supplied_features = [], [], []

    def encode(batch):
        language_inputs.append(batch)
        return batch

    original_decode = model.decode

    def decode(batch, options, audio_features=None):
        transcription_inputs.append(batch)
        supplied_features.append(audio_features)
        return original_decode(batch, options, audio_features=audio_features)

    model.embed_audio = encode
    model.detect_language = lambda features: (None, {"en": 1.0})
    model.decode = decode
    run(seconds, language=None, batch_size=3)
    # Language detection retains mel frames from waveform silence padding.
    # The actual short transcription window pads its log-mel tensor with zeros.
    assert language_inputs[0].shape == transcription_inputs[0].shape
    assert not mx.array_equal(language_inputs[0], transcription_inputs[0]).item()
    assert len(supplied_features) == 1
    assert supplied_features[0] is None
    assert model.encodings == 1


@pytest.mark.parametrize("value", [0.0, 0.25, -0.5])
@pytest.mark.parametrize("without_timestamps", [False, True])
def test_constant_audio_cannot_emit_confident_hallucination(
    transcription, value, without_timestamps
):
    model, run = transcription
    model.answer = lambda *args: {"no_speech_prob": 0.9, "avg_logprob": -0.1}
    result = run(
        29,
        waveform=np.full(29 * 16000, value, np.float32),
        without_timestamps=without_timestamps,
        initial_prompt="names",
    )
    assert result == {"text": "", "segments": [], "language": "en"}
    assert model.calls == []


def test_constant_window_between_speech_preserves_context_and_absolute_seek(
    transcription,
):
    model, run = transcription
    waveform = np.resize(np.array([0.0, 1.0], np.float32), 90 * 16000)
    # Include the STFT context around this middle window in its silent interval.
    waveform[30 * 16000 - 200 : 60 * 16000 + 200] = 0
    result = run(90, waveform=waveform, initial_prompt="names", batch_size=3)
    assert [starts for starts, _ in model.calls] == [[0], [6000]]
    assert [options.prompt for _, options in model.calls] == [[99], [99, 1]]
    assert [segment["seek"] for segment in result["segments"]] == [0, 6000]
    assert result["text"] == "1 3 "


@pytest.mark.parametrize("amplitude", [1e-8, 1e-12])
def test_exact_gate_preserves_arbitrarily_quiet_nonconstant_samples(
    transcription, amplitude
):
    model, run = transcription
    waveform = np.zeros(16000, np.float32)
    waveform[8000] = amplitude
    # Keep the established confidence override for actual varying samples.
    model.answer = lambda *args: {"no_speech_prob": 0.9, "avg_logprob": -0.1}
    assert run(1, waveform=waveform)["text"] == "1 "
    assert len(model.calls) == 1


def test_silent_middle_window_is_omitted_from_independent_batch(transcription):
    model, run = transcription
    waveform = np.resize(np.array([0.0, 1.0], np.float32), 90 * 16000)
    waveform[30 * 16000 - 200 : 60 * 16000 + 200] = 0
    result = run(90, waveform=waveform, batch_size=3, condition_on_previous_text=False)
    assert [starts for starts, _ in model.calls] == [[0, 6000]]
    assert [segment["seek"] for segment in result["segments"]] == [0, 6000]


def test_clip_gate_keeps_stft_context_and_absolute_offsets(transcription):
    model, run = transcription
    waveform = np.zeros(60 * 16000, np.float32)
    # This sample is before the selected clip, but contributes to its first
    # centered STFT frame. The gate must conservatively retain that window.
    waveform[30 * 16000 - 100] = 1e-8
    result = run(60, waveform=waveform, clip_timestamps=[30, 40])
    assert [starts for starts, _ in model.calls] == [[3000]]
    assert result["segments"][0]["start"] == 30
    assert result["segments"][0]["end"] == 40


def test_fully_constant_clip_does_not_transcribe_speech_outside_context(transcription):
    model, run = transcription
    waveform = np.zeros(60 * 16000, np.float32)
    waveform[0] = 1
    result = run(60, waveform=waveform, clip_timestamps=[30, 40])
    assert result == {"text": "", "segments": [], "language": "en"}
    assert model.calls == []


def test_auto_language_is_preserved_when_transcription_windows_are_constant(
    transcription,
):
    model, run = transcription
    model.is_multilingual, model.num_languages = True, 99
    detections = []
    model.embed_audio = lambda batch: batch
    model.detect_language = lambda features: (
        detections.append(features) or None,
        {"fr": 1.0},
    )
    result = run(29, waveform=np.zeros(29 * 16000, np.float32), language=None)
    assert result == {"text": "", "segments": [], "language": "fr"}
    assert len(detections) == 1
    assert model.calls == []


def test_file_waveform_is_decoded_once_before_mel_and_gate(transcription, monkeypatch):
    from lightning_whisper_mlx import audio

    model, run = transcription
    calls = []
    waveform = mx.array([0.0, 1.0] * 8000)
    monkeypatch.setattr(
        audio, "load_audio", lambda path: calls.append(path) or waveform
    )
    run(1, waveform="recording.wav")
    assert calls == ["recording.wav"]
    assert len(model.calls) == 1


def test_transcription_validates_pcm_once_before_prepared_mel(
    transcription, monkeypatch
):
    from lightning_whisper_mlx import audio

    model, _ = transcription
    model.dims = SimpleNamespace(n_mels=80, n_audio_ctx=1500)
    original = audio.prepare_audio
    preparations = []

    def prepare(samples):
        preparations.append(samples)
        return original(samples)

    monkeypatch.setattr(tr, "prepare_audio", prepare)
    monkeypatch.setattr(audio, "prepare_audio", prepare)
    result = tr.transcribe_audio(
        np.zeros(16000, np.float32),
        model_instance=model,
        language="en",
        fp16=False,
        temperature=0.0,
    )
    assert result["text"] == ""
    assert len(preparations) == 1


def test_independent_request_batch_preserves_empty_and_speech_result_order(
    transcription, monkeypatch
):
    model, _ = transcription
    monkeypatch.setattr(
        tr,
        "log_mel_spectrogram_from_prepared",
        lambda audio, **kwargs: mx.array(
            (np.arange(3300, dtype=np.float32) + audio[0].item() * 30000)[:, None]
        ),
    )
    outputs = tr.transcribe_many_audio(
        [
            np.zeros(48000, np.float32),
            np.resize(np.array([1.0, 2.0], np.float32), 48000),
        ],
        model_instance=model,
        language="en",
        fp16=False,
        batch_size=2,
        temperature=0.0,
    )
    assert outputs[0] == {"text": "", "segments": [], "language": "en"}
    assert outputs[1]["text"] == "11 "
    assert [starts for starts, _ in model.calls] == [[30000]]
