"""Backend-neutral text quality and paired timing assessment."""

from __future__ import annotations

import math
from statistics import median, stdev
import unicodedata


def normalize_text(text):
    """NFKC, casefold, replace punctuation with spaces; preserve letters/numbers."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(
            " " if unicodedata.category(c).startswith("P") else c for c in text
        ).split()
    )


def edit_distance(reference, hypothesis):
    """Exact Levenshtein distance using bit-parallel dynamic programming.

    Python integers provide an unbounded bit vector, keeping long transcription
    CER checks practical without a native dependency or a quadratic table.
    """
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    width = len(reference)
    if not width:
        return len(hypothesis)
    equality = {}
    for index, token in enumerate(reference):
        equality[token] = equality.get(token, 0) | (1 << index)
    positive, negative, score = (1 << width) - 1, 0, width
    high = 1 << (width - 1)
    for token in hypothesis:
        equal = equality.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        add = negative | ~(horizontal | positive)
        subtract = positive & horizontal
        score += bool(add & high) - bool(subtract & high)
        add = (add << 1) | 1
        subtract <<= 1
        positive = subtract | ~(vertical | add)
        negative = add & vertical
    return score


def text_quality(reference, hypothesis):
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    words, chars = reference.split(), list(reference.replace(" ", ""))
    word_errors = edit_distance(words, hypothesis.split())
    char_errors = edit_distance(chars, list(hypothesis.replace(" ", "")))
    return {
        "word_errors": word_errors,
        "reference_words": len(words),
        "character_errors": char_errors,
        "reference_characters": len(chars),
        "wer": word_errors / len(words) if words else (0.0 if not hypothesis else None),
        "cer": char_errors / len(chars) if chars else (0.0 if not hypothesis else None),
        "empty_reference_hallucination": not reference and bool(hypothesis),
    }


def summarize_times(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timings must be finite and positive")
    ordered = sorted(values)
    return {
        "samples_seconds": list(values),
        "median_seconds": median(values),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "mean_seconds": sum(values) / len(values),
        "standard_deviation_seconds": stdev(values) if len(values) > 1 else None,
    }


def assess_backend_results(
    results, references=None, *, max_wer_regression=0.0, max_cer_regression=0.0
):
    """Reject quality claims without references and reproducible output stability.

    Reference texts must be one per clip, in input order; empty text is a valid
    silence reference. Cross-backend exact tokens are diagnostic, not a quality gate.
    """
    if len(results) != 2:
        raise ValueError("expected MLX and whisper.cpp results")
    mlx, cpp = results
    first = mlx.outputs[0]
    if references is not None and len(references) != len(first):
        raise ValueError("reference count differs from clip count")
    if any(
        len(output) != len(first) for result in results for output in result.outputs
    ):
        raise ValueError("output clip count changed")
    stable = [
        all(
            [normalize_text(clip["text"]) for clip in output]
            == [normalize_text(clip["text"]) for clip in result.outputs[0]]
            for output in result.outputs
        )
        for result in results
    ]
    per_clip = []
    quality_pass = references is not None and all(stable)
    for i, left in enumerate(first):
        right = cpp.outputs[0][i]
        entry = {
            "index": i,
            "normalized_text_match": normalize_text(left["text"])
            == normalize_text(right["text"]),
            "language_match": left.get("language") == right.get("language"),
            "segment_count": {
                "mlx": len(left.get("segments", [])),
                "whisper_cpp": len(right.get("segments", [])),
            },
        }
        wall = [
            [output[i].get("wall_seconds") for output in result.outputs]
            for result in results
        ]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for values in wall
            for value in values
        ):
            entry["wall_times"] = {
                "mlx": summarize_times(wall[0]),
                "whisper_cpp": summarize_times(wall[1]),
            }
            entry["wall_median_speedup"] = median(wall[1]) / median(wall[0])
            entry["wall_scope"] = (
                "parent API PCM-to-structured-result; cpp IPC included"
            )
        service = [
            [output[i].get("service_seconds") for output in result.outputs]
            for result in results
        ]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for values in service
            for value in values
        ):
            entry["service_times"] = {
                "mlx": summarize_times(service[0]),
                "whisper_cpp": summarize_times(service[1]),
            }
            entry["service_median_speedup"] = median(service[1]) / median(service[0])
            entry["service_scope"] = "engine-to-structured-result; cpp IPC excluded"
        if references is not None:
            qualities = [
                text_quality(references[i], output[i]["text"])
                for result in results
                for output in result.outputs
            ]
            # All repetitions are checked; deterministic stability is not assumed.
            left_quality = qualities[: len(mlx.outputs)]
            right_quality = qualities[len(mlx.outputs) :]
            entry["quality"] = {"mlx": left_quality, "whisper_cpp": right_quality}
            clip_pass = (
                all(
                    normalize_text(output[i]["text"])
                    == normalize_text(result.outputs[0][i]["text"])
                    for result in results
                    for output in result.outputs
                )
                and entry["language_match"]
            )
            if not normalize_text(references[i]):
                # Empty-reference WER/CER have zero denominators. Judge actual
                # insertions/hallucinations instead; a worse C++ baseline must
                # not disqualify an empty, correct MLX transcription.
                clip_pass = clip_pass and not any(
                    q["empty_reference_hallucination"] for q in left_quality
                )
                entry["quality_gate"] = "empty-reference-no-mlx-hallucination"
            else:
                for field, tolerance in (
                    ("wer", max_wer_regression),
                    ("cer", max_cer_regression),
                ):
                    lvalues = [q[field] for q in left_quality]
                    rvalues = [q[field] for q in right_quality]
                    if (
                        any(value is None for value in lvalues + rvalues)
                        or max(lvalues) > min(rvalues) + tolerance
                    ):
                        clip_pass = False
                entry["quality_gate"] = "relative-wer-and-cer"
            entry["quality_pass"] = clip_pass
            quality_pass = quality_pass and clip_pass
        if not entry["language_match"]:
            quality_pass = False
        per_clip.append(entry)
    ratios = [
        right / left
        for left, right in zip(mlx.durations_seconds, cpp.durations_seconds)
    ]
    ratio = cpp.median_seconds / mlx.median_seconds
    return {
        "median_speedup": ratio,
        "paired_round_speedups": ratios,
        "paired_speedup_min": min(ratios),
        "paired_speedup_max": max(ratios),
        "normalized_outputs_stable": {"mlx": stable[0], "whisper_cpp": stable[1]},
        "quality_status": (
            "passed-relative-text-gate"
            if quality_pass
            else ("unqualified-no-references" if references is None else "failed")
        ),
        "quality_limits": {
            "max_wer_regression": max_wer_regression,
            "max_cer_regression": max_cer_regression,
        },
        "timestamps_quality": "unqualified; requires reference timestamps and coverage review",
        "meets_1_2_reference_floor": ratio >= 1.2,
        "speed_claim_qualified": quality_pass,
        "per_clip": per_clip,
        "timings": {
            "mlx": summarize_times(mlx.durations_seconds),
            "whisper_cpp": summarize_times(cpp.durations_seconds),
        },
    }
