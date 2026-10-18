"""Deterministic bounded beam search over the ordinary cached inference path."""

import math

import mlx.core as mx


class BeamSearchDecoder:
    """Maintain independent live beams and completed hypotheses for each audio.

    Only ``beam_size + 1`` token candidates per live row cross the host boundary.
    Cache rows follow selected parents, including when several children share one
    parent. Finished hypotheses never consume a live beam slot.
    """

    def __init__(self, beam_size, eot, inference, patience=None):
        if (
            not isinstance(beam_size, int)
            or isinstance(beam_size, bool)
            or beam_size < 1
        ):
            raise ValueError("beam_size must be a positive integer")
        if not isinstance(eot, int) or isinstance(eot, bool) or eot < 0:
            raise ValueError("eot must be a non-negative integer")
        if patience is None:
            patience = 1.0
        if (
            isinstance(patience, bool)
            or not isinstance(patience, (int, float))
            or not math.isfinite(patience)
            or patience <= 0
        ):
            raise ValueError("patience must be finite and positive")
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.max_candidates = round(beam_size * patience)
        if self.max_candidates < 1:
            raise ValueError(
                "beam_size * patience must round to at least one candidate"
            )
        self.reset()

    def reset(self):
        self.finished_sequences = None
        self._exhausted = None

    @staticmethod
    def _rank(item):
        sequence, (score, _) = item
        return (-score, sequence)

    def _top_candidates(self, logprobs):
        n_vocab = logprobs.shape[-1]
        count = min(self.beam_size + 1, n_vocab)
        selected = mx.argpartition(-logprobs, kth=count - 1, axis=-1)[:, :count]
        cutoff = mx.min(
            mx.take_along_axis(logprobs, selected, axis=-1), axis=-1, keepdims=True
        )
        token_ids = mx.arange(n_vocab)[None, :]
        # argpartition alone arbitrarily chooses among equal boundary scores.
        # All strict winners fit; use token IDs to choose the remaining ties.
        priority = mx.where(
            logprobs > cutoff,
            token_ids - n_vocab,
            mx.where(logprobs == cutoff, token_ids, 2 * n_vocab),
        )
        selected = mx.argpartition(priority, kth=count - 1, axis=-1)[:, :count]
        values = mx.take_along_axis(logprobs, selected, axis=-1)
        mx.eval(selected, values)
        return selected.tolist(), values.tolist()

    def update(self, tokens, logits, sum_logprobs):
        if tokens.ndim != 2 or logits.ndim != 2 or sum_logprobs.ndim != 1:
            raise ValueError(
                "beam update expects rank-2 tokens/logits and rank-1 scores"
            )
        n_rows = tokens.shape[0]
        if (
            n_rows % self.beam_size
            or logits.shape[0] != n_rows
            or sum_logprobs.shape[0] != n_rows
        ):
            raise ValueError(
                "beam rows must match scores/logits and be divisible by beam_size"
            )
        if self.eot >= logits.shape[-1] or not tokens.shape[-1] or not n_rows:
            raise ValueError(
                "beam input requires nonempty tokens and an in-vocabulary eot"
            )
        n_audio = n_rows // self.beam_size
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]
            self._exhausted = [False] * n_audio
        elif len(self.finished_sequences) != n_audio:
            raise ValueError("audio batch changed without resetting beam decoder")
        logits = logits.astype(mx.float32)
        normalizer = mx.logsumexp(logits, axis=-1, keepdims=True)
        logprobs = mx.where(mx.isfinite(normalizer), logits - normalizer, -float("inf"))
        candidate_ids, candidate_probs = self._top_candidates(logprobs)
        histories = tokens.tolist()
        previous_scores = sum_logprobs.tolist()
        next_sequences, next_scores, parents = [], [], []
        for audio_index in range(n_audio):
            start = audio_index * self.beam_size
            finished = self.finished_sequences[audio_index]
            if len(finished) >= self.max_candidates or self._exhausted[audio_index]:
                next_sequences.extend(
                    [
                        history + [self.eot]
                        for history in histories[start : start + self.beam_size]
                    ]
                )
                next_scores.extend(previous_scores[start : start + self.beam_size])
                parents.extend(range(start, start + self.beam_size))
                continue
            candidates = {}
            for row in range(start, start + self.beam_size):
                history = tuple(histories[row])
                previous = previous_scores[row]
                if not math.isfinite(previous):
                    continue
                if history[-1] == self.eot:
                    finished[history] = max(
                        previous, finished.get(history, -float("inf"))
                    )
                    continue
                for token_id, probability in zip(
                    candidate_ids[row], candidate_probs[row]
                ):
                    score = previous + probability
                    if not math.isfinite(score):
                        continue
                    sequence = history + (token_id,)
                    old = candidates.get(sequence)
                    if (
                        old is None
                        or score > old[0]
                        or (score == old[0] and row < old[1])
                    ):
                        candidates[sequence] = (score, row)
            active = []
            for sequence, (score, parent) in sorted(candidates.items(), key=self._rank):
                if sequence[-1] == self.eot:
                    finished[sequence] = max(
                        score, finished.get(sequence, -float("inf"))
                    )
                else:
                    active.append((sequence, score, parent))
                    if len(active) == self.beam_size:
                        break
            # Keep the best completed hypotheses regardless of insertion order.
            retained = sorted(finished.items(), key=lambda item: (-item[1], item[0]))[
                : self.max_candidates
            ]
            finished.clear()
            finished.update(retained)
            if not active and not finished:
                raise ValueError("beam search has no finite continuation")
            if not active:
                self._exhausted[audio_index] = True
                sequence = tuple(histories[start]) + (self.eot,)
                active = [(sequence, -float("inf"), start)]
            for sequence, score, parent in active:
                next_sequences.append(sequence)
                next_scores.append(score)
                parents.append(parent)
            # Maintain fixed batch shape when fewer finite unique continuations
            # exist. Padding beams cannot compete or create finished hypotheses.
            for _ in range(self.beam_size - len(active)):
                next_sequences.append(active[0][0])
                next_scores.append(-float("inf"))
                parents.append(active[0][2])
        self.inference.rearrange_kv_cache(parents)
        complete = all(
            len(finished) >= self.max_candidates or exhausted
            for finished, exhausted in zip(self.finished_sequences, self._exhausted)
        )
        return (
            mx.array(next_sequences, dtype=tokens.dtype),
            complete,
            mx.array(next_scores, dtype=sum_logprobs.dtype),
        )

    def finalize(self, tokens, sum_logprobs):
        if (
            tokens.ndim != 3
            or sum_logprobs.ndim != 2
            or tokens.shape[:2] != sum_logprobs.shape
            or tokens.shape[1] != self.beam_size
        ):
            raise ValueError(
                "beam finalize expects (audio, beam, tokens) and (audio, beam) scores"
            )
        histories, scores = tokens.tolist(), sum_logprobs.tolist()
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in histories]
        if len(self.finished_sequences) != len(histories):
            raise ValueError("audio batch changed without resetting beam decoder")
        batches, batch_scores = [], []
        for finished, live_histories, live_scores in zip(
            self.finished_sequences, histories, scores
        ):
            candidates = dict(finished)
            if len(candidates) < self.beam_size:
                for history, score in sorted(
                    zip(live_histories, live_scores),
                    key=lambda item: (-item[1], tuple(item[0])),
                ):
                    if not math.isfinite(score):
                        continue
                    sequence = tuple(history)
                    if not sequence or sequence[-1] != self.eot:
                        sequence += (self.eot,)
                    candidates[sequence] = max(
                        score, candidates.get(sequence, -float("inf"))
                    )
                    if len(candidates) >= self.beam_size:
                        break
            ordered = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
            if not ordered:
                raise ValueError("beam search produced no finite final hypothesis")
            batches.append([sequence for sequence, _ in ordered])
            batch_scores.append([score for _, score in ordered])
        width = max(len(batch) for batch in batches)
        length = max(len(sequence) for batch in batches for sequence in batch)
        padded = []
        for batch, scores in zip(batches, batch_scores):
            padded_batch = [
                list(sequence) + [self.eot] * (length - len(sequence))
                for sequence in batch
            ]
            while len(padded_batch) < width:
                padded_batch.append(padded_batch[0].copy())
                scores.append(-float("inf"))
            padded.append(padded_batch)
        return mx.array(padded, dtype=tokens.dtype), batch_scores
