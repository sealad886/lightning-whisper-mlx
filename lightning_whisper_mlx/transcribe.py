# Copyright © 2023 Apple Inc.

import sys
import warnings
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import tqdm
import time

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .load_models import load_model, model_artifact_identity
from .mtp import fingerprint_tokenizer, load_mtp_head
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer


def _format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


class ModelHolder:
    model = None
    model_path = None
    model_identity = None

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        identity = (model_artifact_identity(model_path), str(dtype))
        if cls.model is None or identity != cls.model_identity:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
            cls.model_identity = (model_artifact_identity(model_path), str(dtype))
        return cls.model


def transcribe_audio(
    audio: Union[str, np.ndarray, mx.array],
    *,
    path_or_hf_repo: str = "mlx-community/whisper-tiny",
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    batch_size: int = 6,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array]
        The path to the audio file to open, or the audio waveform

    path_or_hf_repo: str
        The localpath to the Whisper model or HF Hub repo with the MLX converted weights.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    temperatures = (
        [temperature] if isinstance(temperature, (int, float)) else list(temperature)
    )
    if not temperatures or any(not np.isfinite(t) or t < 0 for t in temperatures):
        raise ValueError("temperature must contain finite, non-negative values")
    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
    model = ModelHolder.get_model(path_or_hf_repo, dtype)

    assistant = None
    if decode_options.get("speculative_mode") == "assistant":
        assistant = decode_options.get("assistant_model")
        if isinstance(assistant, str):
            assistant = load_model(assistant, dtype=dtype)
            decode_options["assistant_model"] = assistant

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES
    content_duration = content_frames * HOP_LENGTH / SAMPLE_RATE

    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":
            make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
                system_encoding
            )
        else:
            make_safe = lambda x: x

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` decoding option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if assistant is not None:
        if assistant.dims.n_vocab != model.dims.n_vocab:
            raise ValueError("assistant and target models must share vocabulary size")
        if assistant.dims.n_mels != model.dims.n_mels:
            raise ValueError("assistant and target models must share mel bin count")
        if assistant.dims.n_text_ctx < model.dims.n_text_ctx:
            raise ValueError("assistant text context must cover target context")
        assistant_tokenizer = get_tokenizer(
            assistant.is_multilingual,
            num_languages=assistant.num_languages,
            language=language,
            task=task,
        )
        if fingerprint_tokenizer(assistant_tokenizer) != fingerprint_tokenizer(
            tokenizer
        ):
            raise ValueError("assistant and target tokenizers are incompatible")
    elif decode_options.get("speculative_mode") == "mtp":
        checkpoint = decode_options.get("mtp_checkpoint")
        if not checkpoint:
            raise ValueError("mtp mode requires mtp_checkpoint")
        head, _ = load_mtp_head(
            checkpoint,
            decoder_hidden_size=model.dims.n_text_state,
            vocabulary_size=model.dims.n_vocab,
            base_model_fingerprint=getattr(model, "_checkpoint_fingerprint", ""),
            tokenizer_fingerprint=fingerprint_tokenizer(tokenizer),
        )
        decode_options["mtp_head"] = head

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    if any(not np.isfinite(ts) or ts < 0 for ts in clip_timestamps):
        raise ValueError("clip timestamps must be finite and non-negative")
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))
    if any(start > end for start, end in seek_clips):
        raise ValueError("clip start must not exceed clip end")
    seek_clips = [
        (min(start, content_frames), min(end, content_frames))
        for start, end in seek_clips
    ]

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def decode_process(segment_batch, t):
        kwargs = {**decode_options}
        if t > 0:
            kwargs.pop("beam_size", None)
            kwargs.pop("patience", None)
        else:
            kwargs.pop("best_of", None)
        if t != 0 and kwargs.get("speculative_mode") != "none":
            kwargs.update(
                speculative_mode="none",
                assistant_model=None,
                assistant_quant=None,
                mtp_checkpoint=None,
                mtp_head=None,
            )
        options = DecodingOptions(**kwargs, temperature=t)
        decode_results = model.decode(segment_batch, options)
        return decode_results

    def decode_with_fallback(segment_batch: mx.array) -> DecodingResult:
        final_decode = [None] * segment_batch.shape[0]
        pending = list(range(segment_batch.shape[0]))
        for t in temperatures:
            results = decode_process(segment_batch[pending], t)
            retry = []
            for index, result in zip(pending, results):
                final_decode[index] = result
                failed = (
                    compression_ratio_threshold is not None
                    and result.compression_ratio > compression_ratio_threshold
                ) or (
                    logprob_threshold is not None
                    and result.avg_logprob < logprob_threshold
                )
                silent = (
                    no_speech_threshold is not None
                    and result.no_speech_prob > no_speech_threshold
                    and logprob_threshold is not None
                    and result.avg_logprob < logprob_threshold
                )
                if failed and not silent:
                    retry.append(index)
            pending = retry
            if not pending:
                break
        return final_decode

    input_stride = N_FRAMES // model.dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *,
        current_seek: int,
        start: float,
        end: float,
        tokens: mx.array,
        result: DecodingResult,
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": current_seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    last_speech_timestamp = 0.0

    def format_output(tokens, res, current_seek, segment_size, mel_segment):
        nonlocal last_speech_timestamp
        previous_seek = current_seek
        seek = current_seek
        time_offset = current_seek * HOP_LENGTH / SAMPLE_RATE
        segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
        window_end_time = time_offset + segment_duration
        current_segments = []

        if no_speech_threshold is not None:
            should_skip = res.no_speech_prob > no_speech_threshold
            if logprob_threshold is not None and res.avg_logprob > logprob_threshold:
                should_skip = False

            if should_skip:
                seek += segment_size
                return current_segments, seek

        def word_anomaly_score(word: dict) -> float:
            probability = word.get("probability", 0.0)
            duration = word["end"] - word["start"]
            score = 0.0
            if probability < 0.15:
                score += 1.0
            if duration < 0.133:
                score += (0.133 - duration) * 15
            if duration > 2.0:
                score += duration - 2.0
            return score

        def is_segment_anomaly(segment: Optional[dict]) -> bool:
            if segment is None or not segment["words"]:
                return False
            words = [w for w in segment["words"] if w["word"] not in punctuation]
            words = words[:8]
            score = sum(word_anomaly_score(w) for w in words)
            return score >= 3 or score + 0.01 >= len(words)

        def next_words_segment(segments: List[dict]) -> Optional[dict]:
            return next((s for s in segments if s["words"]), None)

        timestamp_tokens = tokens >= tokenizer.timestamp_begin
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [
            False,
            True,
        ]

        consecutive = np.where(
            np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
        )[0]
        consecutive += 1
        if len(consecutive) > 0:
            slices = consecutive.tolist()
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_pos = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_pos = sliced_tokens[-1].item() - tokenizer.timestamp_begin
                current_segments.append(
                    new_segment(
                        current_seek=current_seek,
                        start=time_offset + start_timestamp_pos * time_precision,
                        end=time_offset + end_timestamp_pos * time_precision,
                        tokens=sliced_tokens,
                        result=res,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                seek += segment_size
            else:
                last_timestamp_pos = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_pos * input_stride
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero()[0]]
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != tokenizer.timestamp_begin
            ):
                last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
                duration = last_timestamp_pos * time_precision

            current_segments.append(
                new_segment(
                    current_seek=current_seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                    result=res,
                )
            )
            seek += segment_size

        if word_timestamps:
            add_word_timestamps(
                segments=current_segments,
                model=model,
                tokenizer=tokenizer,
                mel=mel_segment,
                num_frames=segment_size,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                last_speech_timestamp=last_speech_timestamp,
            )

            if not single_timestamp_ending:
                last_word_end = _get_end(current_segments)
                if last_word_end is not None and last_word_end > time_offset:
                    seek = round(last_word_end * FRAMES_PER_SECOND)

            # skip silence before possible hallucinations
            if hallucination_silence_threshold is not None:
                threshold = hallucination_silence_threshold
                if not single_timestamp_ending:
                    last_word_end = _get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        remaining_duration = window_end_time - last_word_end
                        if remaining_duration > threshold:
                            seek = round(last_word_end * FRAMES_PER_SECOND)
                        else:
                            seek = previous_seek + segment_size

                # if first segment might be a hallucination, skip leading silence
                first_segment = next_words_segment(current_segments)
                if first_segment is not None and is_segment_anomaly(first_segment):
                    gap = first_segment["start"] - time_offset
                    if gap > threshold:
                        seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                        return [], seek

                # skip silence before any possible hallucination that is surrounded
                # by silence or more hallucinations
                hal_last_end = last_speech_timestamp
                for si in range(len(current_segments)):
                    segment = current_segments[si]
                    if not segment["words"]:
                        continue
                    if is_segment_anomaly(segment):
                        next_segment = next_words_segment(current_segments[si + 1 :])
                        if next_segment is not None:
                            hal_next_start = next_segment["words"][0]["start"]
                        else:
                            hal_next_start = time_offset + segment_duration
                        silence_before = (
                            segment["start"] - hal_last_end > threshold
                            or segment["start"] < threshold
                            or segment["start"] - time_offset < 2.0
                        )
                        silence_after = (
                            hal_next_start - segment["end"] > threshold
                            or is_segment_anomaly(next_segment)
                            or window_end_time - segment["end"] < 2.0
                        )
                        if silence_before and silence_after:
                            seek = round(
                                max(time_offset + 1, segment["start"])
                                * FRAMES_PER_SECOND
                            )
                            if content_duration - segment["end"] < threshold:
                                seek = content_frames
                            current_segments[si:] = []
                            break
                    hal_last_end = segment["end"]

            last_word_end = _get_end(current_segments)
            if last_word_end is not None:
                last_speech_timestamp = last_word_end

        for i, segment in enumerate(current_segments):
            if segment["start"] == segment["end"] or segment["text"].strip() == "":
                segment["text"] = ""
                segment["tokens"] = []
                segment["words"] = []

        return current_segments, seek

    # Conditioned windows depend on the previous decoded text; alignment can also
    # change the next seek. Only independent windows may be decoded in a batch.
    window_batch_size = (
        1 if condition_on_previous_text or word_timestamps else batch_size
    )
    for clip_start, clip_end in seek_clips:
        seek = clip_start
        while seek < clip_end:
            chunks = []
            candidate_seek = seek
            # An initial prompt belongs only to the first window, even when
            # subsequent independent windows can share a batch.
            width = (
                1
                if initial_prompt_tokens and prompt_reset_since == 0
                else window_batch_size
            )
            for _ in range(width):
                if candidate_seek >= clip_end:
                    break
                size = min(
                    N_FRAMES, content_frames - candidate_seek, clip_end - candidate_seek
                )
                mel_segment = pad_or_trim(
                    mel[candidate_seek : candidate_seek + size], N_FRAMES, axis=-2
                ).astype(dtype)
                chunks.append((candidate_seek, size, mel_segment))
                candidate_seek += size

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            results = decode_with_fallback(mx.stack([chunk[2] for chunk in chunks]))
            for (current_seek, size, mel_segment), res in zip(chunks, results):
                tokens = np.array(res.tokens)
                current_segments, next_seek = format_output(
                    tokens, res, current_seek, size, mel_segment
                )
                for segment in current_segments:
                    if verbose:
                        line = f"[{_format_timestamp(segment['start'])} --> {_format_timestamp(segment['end'])}] {segment['text']}"
                        print(make_safe(line))
                    all_segments.append({"id": len(all_segments), **segment})
                    all_tokens.extend(segment["tokens"])
                if not condition_on_previous_text or res.temperature > 0.5:
                    prompt_reset_since = len(all_tokens)
                seek = min(clip_end, max(current_seek + 1, next_seek))
                # An unfinished timestamp tail invalidates later predecoded
                # windows: resume at its last completed timestamp instead.
                if seek != current_seek + size:
                    break

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
