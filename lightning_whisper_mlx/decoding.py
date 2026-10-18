# Copyright © 2023 Apple Inc.

import zlib
from functools import lru_cache
from types import SimpleNamespace
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
import numpy as np

from .audio import CHUNK_LENGTH
from .beam import BeamSearchDecoder
from .speculative import (
    AssistantSpeculativeDecoder,
    MTPSpeculativeDecoder,
    SpeculativeOptions,
    validate_speculative_options,
)
from .tokenizer import Tokenizer, get_tokenizer

if TYPE_CHECKING:
    from .whisper import Whisper


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def detect_language(
    model: "Whisper", mel: mx.array, tokenizer: Tokenizer = None
) -> Tuple[mx.array, Union[Dict[str, float], List[Dict[str, float]]]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : mx.array, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.embed_audio(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mx.array(mask)
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation
    pipeline: bool = False  # optional bounded one-step scheduling experiment

    # lossless speculative decoding; disabled by default
    speculative_mode: str = "none"
    lookahead: int = 2
    assistant_model: object = None
    assistant_quant: Optional[str] = None
    mtp_checkpoint: Optional[str] = None
    mtp_head: object = None
    speculative_batch_policy: str = "lockstep"


@dataclass(frozen=True)
class DecodingResult:
    audio_features: mx.array
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None
        self.last_hidden = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Perform a forward pass on the decoder and return per-token logits"""
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.forward_block(tokens, audio_features)

    def prefill(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Run the initial prompt without reducing it to its final token."""
        if self.kv_cache is not None:
            raise RuntimeError("prefill requires an empty inference cache")
        return self.forward_block(tokens, audio_features)

    def forward_one(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Decode one token against the current cache."""
        if tokens.shape[-1] != 1:
            raise ValueError("forward_one expects exactly one token")
        return self.forward_block(tokens, audio_features)

    def forward_block(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Queue cached tokens with the same arithmetic as sequential decoding.

        Multi-row fp16 matmuls produce different logits/cache values from the
        single-row target path. Queue single-token graphs without intermediate
        evaluation so proposal verification preserves the target's arithmetic.
        Initial prompt prefill remains one causal decoder call.
        """
        if self.kv_cache is not None and tokens.shape[-1] > 1:
            return mx.concatenate(
                [
                    self.forward_one(tokens[:, index : index + 1], audio_features)
                    for index in range(tokens.shape[-1])
                ],
                axis=1,
            )
        compiled = getattr(self.model, "_compiled_decoder", None)
        if compiled is not None and self.kv_cache is not None:
            offset = self.cache_length
            end = offset + tokens.shape[-1]
            for cache, _ in self.kv_cache:
                cache.ensure_capacity(end)
            logits, state, self.last_hidden = compiled(
                tokens, audio_features, self.cache_state, mx.array(offset)
            )
            for (cache, _), (buffers, _) in zip(self.kv_cache, state):
                cache.keys, cache.values = buffers
                cache.offset = end
            return logits.astype(mx.float32)
        logits, self.kv_cache, _, self.last_hidden = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache, return_hidden=True
        )
        return logits.astype(mx.float32)

    def sample_next(self, tokens, audio_features, decoder, sum_logprobs, max_steps=1):
        """Run bounded sequential ordinary steps and sampling in one graph.

        The token history includes exactly one token not yet in the cache. The
        first prompt evaluation and its SOT probabilities stay on the ordinary
        path; speculation continues to use the logits/hidden-state interface.
        """
        length = decoder._prepare_history(tokens)
        offset = self.cache_length
        if length != offset + 1 or self.kv_cache is None:
            raise ValueError("joint sampling requires one uncached history token")
        for owner, _ in self.kv_cache:
            owner.ensure_capacity(offset + 1)
        # Finish at a capacity boundary before growing. Growing early would make
        # the first step attend across a different padded width than one-step
        # execution, potentially changing its floating-point arithmetic.
        steps = min(
            max_steps,
            self.kv_cache[0][0].keys.shape[1] - offset,
            decoder.max_tokens - length,
        )
        sampling_key = decoder._sampling_key
        key = (sampling_key, steps)
        cache = self.model._joint_decoder_cache
        compiled = cache.get(key)
        if compiled is None:
            filters = tuple(_filter_from_key(item) for item in sampling_key[2])
            model_decoder = self.model.decoder
            eot = decoder.eot

            def step(token, features, state, offset, history, scores, length):
                for index in range(steps):
                    logits, state, _ = model_decoder(
                        token, features, kv_cache=state, offset=offset + index
                    )
                    history, completed, scores = _greedy_sampling_step(
                        logits.astype(mx.float32)[:, -1],
                        history,
                        scores,
                        length + index,
                        filters,
                        eot,
                    )
                    token = mx.take(history, (length + index).reshape(1), axis=1)
                return history, state, completed, scores

            compiled = mx.compile(step)
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            cache[key] = compiled
        end = offset + steps
        history, state, completed, sum_logprobs = compiled(
            tokens[:, -1:],
            audio_features,
            self.cache_state,
            mx.array(offset),
            decoder._tokens,
            sum_logprobs,
            mx.array(length),
        )
        for (owner, _), (buffers, _) in zip(self.kv_cache, state):
            owner.keys, owner.values = buffers
            owner.offset = end
        self.last_hidden = None
        decoder._tokens = history
        decoder._length = length + steps
        return history[:, : length + steps], completed, sum_logprobs

    @property
    def cache_length(self) -> int:
        if not self.kv_cache or self.kv_cache[0] is None:
            return 0
        self_kv = self.kv_cache[0][0]
        return 0 if self_kv is None else self_kv.offset

    def truncate(self, length: int):
        """Drop uncommitted self-attention cache entries."""
        if length < 0 or length > self.cache_length:
            raise ValueError("cache truncation length is outside the current cache")
        if self.kv_cache is None:
            if length:
                raise ValueError("cannot retain entries in an empty cache")
            return

        for layer_cache in self.kv_cache:
            layer_cache[0].truncate(length)

    @property
    def cache_state(self):
        return [(kv.state, cross) for kv, cross in self.kv_cache or []]

    def rearrange_kv_cache(self, source_indices):
        """Update the key-value cache according to the updated beams"""
        # update the key/value cache to contain the selected sequences
        if source_indices != list(range(len(source_indices))):
            reordered = []
            for kv, cross in self.kv_cache:
                kv.reorder(source_indices)
                reordered.append((kv, tuple(x[source_indices] for x in cross)))
            self.kv_cache = reordered

    def reset(self):
        self.kv_cache = None
        self.last_hidden = None


class SequenceRanker:
    def rank(
        self, tokens: List[List[mx.array]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[List[int]]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = max(1, length)
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : mx.array, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : mx.array, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        sum_logprobs: mx.array, shape = (n_batch)
            updated cumulative log probabilities for each sequence

        """
        raise NotImplementedError

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> Tuple[Sequence[Sequence[mx.array]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : mx.array, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : mx.array, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[mx.array]], length = n_audio
            sequence of mx.arrays containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


def _greedy_transition(logits, last_tokens, sum_logprobs, temperature, eot):
    """Shared numerical selection/scoring semantics for eager and compiled paths."""
    next_tokens = (
        logits.argmax(axis=-1)
        if temperature == 0
        else mx.random.categorical(logits=logits / temperature)
    )
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    current = logprobs[mx.arange(logprobs.shape[0]), next_tokens]
    ended = last_tokens == eot
    # An ended row can have every logit filtered. Do not multiply NaN/-inf by
    # zero: its accumulated probability must remain unchanged.
    scores = sum_logprobs + mx.where(ended, 0.0, current)
    next_tokens = mx.where(ended, eot, next_tokens)
    return next_tokens, mx.all(next_tokens == eot), scores


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int, max_tokens: int = 449):
        self.temperature = temperature
        self.eot = eot
        self.max_tokens = max_tokens
        self._compiled_sampler = None
        self._sampling_key = None
        self.reset()

    def reset(self):
        self._tokens = None
        self._length = None

    @property
    def uses_compiled_filters(self):
        return self._compiled_sampler is not None

    def configure_sampling(self, filters):
        if self.temperature != 0:
            raise ValueError("compiled sampling requires temperature=0")
        self._sampling_key = (
            self.eot,
            self.max_tokens,
            tuple(f.cache_key for f in filters),
        )
        self._compiled_sampler = _compiled_greedy_sampler(*self._sampling_key)

    def _prepare_history(self, tokens):
        length = tokens.shape[-1]
        if tokens.ndim != 2 or not length or length >= self.max_tokens:
            raise ValueError("token history is empty or exceeds decoder context")
        if self._length is not None and self._length != length:
            raise ValueError("token history length changed without resetting decoder")
        if self._tokens is None:
            self._tokens = mx.zeros(
                (tokens.shape[0], self.max_tokens), dtype=tokens.dtype
            )
            self._tokens = mx.slice_update(self._tokens, tokens, mx.array(0), axes=(1,))
        elif self._tokens.shape[0] != tokens.shape[0]:
            raise ValueError("token batch changed without resetting decoder")
        return length

    def update(self, tokens, logits, sum_logprobs):
        length = self._prepare_history(tokens)
        if self._compiled_sampler is not None:
            self._tokens, completed, sum_logprobs = self._compiled_sampler(
                logits, self._tokens, sum_logprobs, mx.array(length)
            )
        else:
            next_tokens, completed, sum_logprobs = _greedy_transition(
                logits, tokens[:, -1], sum_logprobs, self.temperature, self.eot
            )
            self._tokens[:, length : length + 1] = next_tokens[:, None]
        self._length = length + 1
        return self._tokens[:, : length + 1], completed, sum_logprobs

    def finalize(self, tokens, sum_logprobs):
        tokens = mx.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class LogitFilter:
    def apply(self, logits, tokens, logical_length=None):
        """Filter logits using only token positions below logical_length."""
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin, n_vocab):
        self.sample_begin = sample_begin
        space_tokens = tuple(tokenizer.encode(" "))
        self.cache_key = ("blank", sample_begin, n_vocab, space_tokens, tokenizer.eot)
        mask = np.zeros(n_vocab, np.float32)
        mask[list(space_tokens) + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits, tokens, logical_length=None):
        length = tokens.shape[1] if logical_length is None else logical_length
        return mx.where(length == self.sample_begin, logits + self.mask, logits)


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens, n_vocab):
        suppressed = tuple(sorted(set(suppress_tokens)))
        self.cache_key = ("tokens", suppressed, n_vocab)
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppressed)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits, tokens, logical_length=None):
        return logits + self.mask


class ApplyTimestampRules(LogitFilter):
    def __init__(self, tokenizer, sample_begin, max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index
        self.cache_key = (
            "timestamps",
            sample_begin,
            max_initial_timestamp_index,
            tokenizer.eot,
            tokenizer.no_timestamps,
            tokenizer.timestamp_begin,
        )

    def apply(self, logits, tokens, logical_length=None):
        length = tokens.shape[1] if logical_length is None else logical_length
        if isinstance(length, int) and not 0 < length <= tokens.shape[1]:
            raise ValueError("logical token length is outside the supplied buffer")
        length = mx.array(length)
        token_ids = mx.arange(logits.shape[-1])[None, :]
        forbidden = mx.zeros(logits.shape, dtype=mx.bool_)
        if self.tokenizer.no_timestamps is not None:
            forbidden = forbidden | (token_ids == self.tokenizer.no_timestamps)
        sampled_length = length - self.sample_begin
        positions = mx.arange(tokens.shape[1])[None, :]
        valid = (positions >= self.sample_begin) & (positions < length)
        is_timestamp = valid & (tokens >= self.tokenizer.timestamp_begin)
        last_token = mx.take(tokens, mx.array(length - 1).reshape(1), axis=1)
        penultimate_token = mx.take(
            tokens, mx.maximum(mx.array(length - 2), 0).reshape(1), axis=1
        )
        has_samples = sampled_length > 0
        last_was_timestamp = has_samples & (
            last_token >= self.tokenizer.timestamp_begin
        )
        penultimate_was_timestamp = (sampled_length < 2) | (
            penultimate_token >= self.tokenizer.timestamp_begin
        )
        forbidden = (
            forbidden
            | (
                has_samples
                & last_was_timestamp
                & penultimate_was_timestamp
                & (token_ids >= self.tokenizer.timestamp_begin)
            )
            | (
                has_samples
                & last_was_timestamp
                & ~penultimate_was_timestamp
                & (token_ids < self.tokenizer.eot)
            )
        )
        last_position = mx.max(
            mx.where(is_timestamp, positions, -1), axis=-1, keepdims=True
        )
        last_timestamp = mx.take_along_axis(
            tokens, mx.maximum(last_position, 0), axis=-1
        )
        lower_bound = last_timestamp + (~last_was_timestamp | penultimate_was_timestamp)
        forbidden = forbidden | (
            (last_position >= 0)
            & (token_ids >= self.tokenizer.timestamp_begin)
            & (token_ids < lower_bound)
        )
        initial = length == self.sample_begin
        forbidden = forbidden | (initial & (token_ids < self.tokenizer.timestamp_begin))
        if self.max_initial_timestamp_index is not None:
            last_allowed = (
                self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            )
            forbidden = forbidden | (initial & (token_ids > last_allowed))
        logits = mx.where(forbidden, -mx.inf, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        timestamp_logprob = mx.logsumexp(
            logprobs[:, self.tokenizer.timestamp_begin :], axis=-1, keepdims=True
        )
        max_text_logprob = mx.max(
            logprobs[:, : self.tokenizer.timestamp_begin], axis=-1, keepdims=True
        )
        return mx.where(
            (timestamp_logprob > max_text_logprob)
            & (token_ids < self.tokenizer.timestamp_begin),
            -mx.inf,
            logits,
        )


def _filter_from_key(key):
    kind, *values = key
    if kind == "tokens":
        return SuppressTokens(*values)
    if kind == "blank":
        sample_begin, n_vocab, space_tokens, eot = values
        tokenizer = SimpleNamespace(encode=lambda _: list(space_tokens), eot=eot)
        return SuppressBlank(tokenizer, sample_begin, n_vocab)
    if kind == "timestamps":
        sample_begin, max_initial, eot, no_timestamps, timestamp_begin = values
        tokenizer = SimpleNamespace(
            eot=eot, no_timestamps=no_timestamps, timestamp_begin=timestamp_begin
        )
        return ApplyTimestampRules(tokenizer, sample_begin, max_initial)
    raise ValueError(f"unsupported compiled logit filter: {kind}")


@lru_cache(maxsize=16)
def _compiled_greedy_sampler(eot, capacity, filter_keys):
    filters = tuple(_filter_from_key(key) for key in filter_keys)

    def step(logits, buffer, scores, length):
        return _greedy_sampling_step(logits, buffer, scores, length, filters, eot)

    return mx.compile(step)


def _greedy_sampling_step(logits, buffer, scores, length, filters, eot):
    """Canonical pure transition shared by separate and joint compiled graphs."""
    for logit_filter in filters:
        logits = logit_filter.apply(logits, buffer, logical_length=length)
    last = mx.take(buffer, (length - 1).reshape(1), axis=1).reshape(buffer.shape[0])
    next_tokens, completed, scores = _greedy_transition(logits, last, scores, 0, eot)
    buffer = mx.slice_update(buffer, next_tokens[:, None], length.reshape(1), axes=(1,))
    return buffer, completed, scores


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(
                options.temperature, tokenizer.eot, self.n_ctx + 1
            )

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin, model.dims.n_vocab)
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens(), model.dims.n_vocab)
            )
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp is not None:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

        if (
            getattr(model, "_compiled_decoder", None) is not None
            and options.temperature == 0
            and options.beam_size is None
        ):
            self.decoder.configure_sampling(self.logit_filters)

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.temperature != 0:
            raise ValueError("beam search requires temperature=0")
        if options.best_of is not None and (
            not isinstance(options.best_of, int) or options.best_of < 1
        ):
            raise ValueError("best_of must be a positive integer")
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        validate_speculative_options(
            SpeculativeOptions(
                speculative_mode=options.speculative_mode,
                lookahead=options.lookahead,
                assistant_model=options.assistant_model,
                assistant_quant=options.assistant_quant,
                mtp_checkpoint=options.mtp_checkpoint,
                speculative_batch_policy=options.speculative_batch_policy,
            ),
            temperature=options.temperature,
        )

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array):
        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.embed_audio(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: mx.array, tokens: np.array):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                # write language tokens
                tokens[:, self.sot_index + 1] = np.array(lang_tokens)

        return languages, lang_probs

    def _main_loop(
        self,
        audio_features: mx.array,
        tokens: mx.array,
        assistant_audio_features: Optional[mx.array] = None,
    ):
        if self.options.speculative_mode == "mtp":
            if tokens.shape[0] != 1:
                raise ValueError(
                    "MTP speculative decoding currently requires batch_size=1"
                )
            if self.options.mtp_head is None:
                raise ValueError("mtp mode requires a loaded MTP head")
            prefix = tokens[0].tolist()
            speculative = MTPSpeculativeDecoder(
                model=self.model,
                head=self.options.mtp_head,
                audio_features=audio_features,
                prefix=prefix,
                lookahead=self.options.lookahead,
                eot=self.tokenizer.eot,
                logit_filters=self.logit_filters,
            )
            no_speech = [np.nan]
            if self.tokenizer.no_speech is not None:
                no_speech = [
                    float(
                        mx.softmax(
                            speculative.verifier.next_logits[:, self.sot_index], axis=-1
                        )[0, self.tokenizer.no_speech].item()
                    )
                ]
            generated, sum_logprob = speculative.decode(self.sample_len)
            return (
                mx.array([prefix + generated]),
                mx.array([sum_logprob]),
                no_speech,
            )

        if self.options.speculative_mode == "assistant":
            if tokens.shape[0] != 1:
                raise ValueError(
                    "assistant speculative decoding currently requires batch_size=1"
                )
            if assistant_audio_features is None:
                raise ValueError("assistant audio features are required")
            prefix = tokens[0].tolist()
            speculative = AssistantSpeculativeDecoder(
                target_model=self.model,
                assistant_model=self.options.assistant_model,
                audio_features=audio_features,
                assistant_audio_features=assistant_audio_features,
                prefix=prefix,
                lookahead=self.options.lookahead,
                eot=self.tokenizer.eot,
                logit_filters=self.logit_filters,
            )
            no_speech = [np.nan]
            if self.tokenizer.no_speech is not None:
                no_speech = [
                    float(
                        mx.softmax(
                            speculative.verifier.next_logits[:, self.sot_index], axis=-1
                        )[0, self.tokenizer.no_speech].item()
                    )
                ]
            generated, sum_logprob = speculative.decode(self.sample_len)
            output = mx.array([prefix + generated])
            return output, mx.array([sum_logprob]), no_speech

        n_batch = tokens.shape[0]
        sum_logprobs: mx.array = mx.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch
        joint = (
            getattr(self.model, "_joint_decoding", False)
            and getattr(self.decoder, "uses_compiled_filters", False)
            and not self.options.pipeline
        )

        try:
            logits = self.inference.logits(tokens, audio_features)
            i = 0
            while i < self.sample_len:
                previous_length = tokens.shape[-1]
                if i and joint:
                    tokens, completed, sum_logprobs = self.inference.sample_next(
                        tokens,
                        audio_features,
                        self.decoder,
                        sum_logprobs,
                        max_steps=min(
                            self.model._decode_chunk_size,
                            self.sample_len - i,
                            self.n_ctx + 1 - previous_length,
                        ),
                    )
                else:
                    if i == 0 and self.tokenizer.no_speech is not None:
                        probs_at_sot = mx.softmax(
                            logits[:, self.sot_index].astype(mx.float32), axis=-1
                        )
                        no_speech_probs = probs_at_sot[
                            :, self.tokenizer.no_speech
                        ].tolist()
                    logits = logits[:, -1]
                    if not getattr(self.decoder, "uses_compiled_filters", False):
                        for logit_filter in self.logit_filters:
                            logits = logit_filter.apply(logits, tokens)
                    tokens, completed, sum_logprobs = self.decoder.update(
                        tokens, logits, sum_logprobs
                    )

                i += tokens.shape[-1] - previous_length
                has_next = i < self.sample_len and tokens.shape[-1] <= self.n_ctx
                next_logits = None
                if has_next and self.options.pipeline:
                    # Schedule bounded work while Python observes the previous
                    # completion flag. EOT rows are already masked by the decoder.
                    next_logits = self.inference.logits(tokens, audio_features)
                    mx.async_eval(next_logits, sum_logprobs, tokens)
                if completed or not has_next:
                    break
                if joint:
                    logits = None
                else:
                    logits = (
                        self.inference.logits(tokens, audio_features)
                        if next_logits is None
                        else next_logits
                    )
        finally:
            self.inference.reset()

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: mx.array, audio_features=None) -> List[DecodingResult]:
        self.decoder.reset()
        if self.options.fp16:
            mel = mel.astype(mx.float16)
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        mel = mx.repeat(mel, repeats=1, axis=0)
        n_audio *= 1

        assistant_audio_features = None
        if self.options.speculative_mode == "assistant":
            assistant_audio_features = self.options.assistant_model.encoder(mel)
        audio_features = self._get_audio_features(
            mel if audio_features is None else audio_features
        )
        tokens: np.array = np.array(self.initial_tokens)
        tokens = np.broadcast_to(tokens, (n_audio, len(self.initial_tokens))).copy()

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat tokens by the group size, for beam search or best-of-n sampling
        tokens = mx.array(tokens)
        if self.n_group > 1:
            tokens = mx.repeat(tokens, self.n_group, axis=0)
            audio_features = mx.repeat(audio_features, self.n_group, axis=0)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(
            audio_features, tokens, assistant_audio_features
        )

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = tokens[..., self.sample_begin :].tolist()
        tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    *,
    audio_features=None,
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: mx.array, shape = (80, 3000) or (*, 80, 3000)
        An array containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    if audio_features is not None and single and audio_features.ndim == 2:
        audio_features = audio_features[None]
    result = DecodingTask(model, options).run(mel, audio_features=audio_features)
    return result[0] if single else result
