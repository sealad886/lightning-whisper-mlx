"""Lossless greedy speculative decoding primitives."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import mlx.core as mx


@dataclass(frozen=True)
class SpeculativeOptions:
    speculative_mode: str = "none"
    lookahead: int = 2
    assistant_model: object = None
    assistant_quant: Optional[str] = None
    mtp_checkpoint: Optional[str] = None
    speculative_batch_policy: str = "lockstep"


@dataclass(frozen=True)
class VerificationResult:
    tokens: List[int]
    accepted: int
    rejected: int
    bonus_tokens: int
    token_logprobs: List[float]
    sum_logprob: float


class AssistantProposer:
    """Greedy assistant proposal generator with cache synchronization."""

    def __init__(
        self, model, audio_features: mx.array, prefix: mx.array, lookahead: int
    ):
        from .decoding import Inference

        self.inference = Inference(model, initial_token_length=prefix.shape[-1])
        self.audio_features = audio_features
        self.lookahead = lookahead
        self._next_logits = self.inference.prefill(prefix, audio_features)
        self._prefix_cache_length = self.inference.cache_length

    @property
    def cache_length(self) -> int:
        return self.inference.cache_length

    def propose(self, max_tokens: Optional[int] = None) -> List[int]:
        proposals: List[int] = []
        logits = self._next_logits[:, -1]
        width = min(
            self.lookahead,
            self.inference.model.dims.n_text_ctx - self.cache_length,
        )
        if max_tokens is not None:
            width = min(width, max_tokens)
        for _ in range(width):
            token = int(mx.argmax(logits, axis=-1)[0].item())
            proposals.append(token)
            self._next_logits = self.inference.forward_one(
                mx.array([[token]]), self.audio_features
            )
            logits = self._next_logits[:, -1]
        return proposals

    def synchronize(self, accepted: int, committed_tokens: Sequence[int]):
        self.inference.truncate(self._prefix_cache_length + accepted)
        for token in committed_tokens[accepted:]:
            if self.cache_length == self.inference.model.dims.n_text_ctx:
                break
            self._next_logits = self.inference.forward_one(
                mx.array([[token]]), self.audio_features
            )
        self._prefix_cache_length = self.inference.cache_length


class CachedGreedyVerifier:
    """Verify proposals while committing only the target-accepted cache prefix."""

    def __init__(
        self,
        *,
        inference,
        audio_features: mx.array,
        prefix: Sequence[int],
        next_logits: mx.array,
        eot: int,
        logit_filters: Optional[Sequence[object]] = None,
    ):
        self.inference = inference
        self.audio_features = audio_features
        self.prefix = list(prefix)
        self.next_logits = next_logits
        self.eot = eot
        self.logit_filters = tuple(logit_filters or ())

    @property
    def cache_length(self) -> int:
        return self.inference.cache_length

    @property
    def remaining_tokens(self) -> int:
        # A full cache can still predict one final token without replaying it.
        return max(0, self.inference.model.dims.n_text_ctx - len(self.prefix) + 1)

    def proposal_limit(self, max_tokens: int) -> int:
        return max(0, min(max_tokens, self.remaining_tokens) - 1)

    def verify(
        self, proposals: Sequence[int], max_tokens: Optional[int] = None
    ) -> VerificationResult:
        budget = self.remaining_tokens
        if max_tokens is not None:
            budget = min(budget, max_tokens)
        if budget <= 0:
            raise ValueError("verification requires remaining token capacity")
        proposals = list(proposals[: self.proposal_limit(budget)])

        committed_length = self.inference.cache_length
        target_logits = [self.next_logits[:, -1]]
        if proposals:
            block_logits = self.inference.forward_block(
                mx.array([proposals]), self.audio_features
            )
            target_logits.extend(
                block_logits[0, index] for index in range(len(proposals))
            )
        result = verify_greedy_proposals(
            prefix=self.prefix,
            proposals=proposals,
            target_logits=target_logits,
            eot=self.eot,
            logit_filters=self.logit_filters,
        )

        self.inference.truncate(committed_length + result.accepted)
        replay = result.tokens[result.accepted :]
        if len(result.tokens) == budget:
            replay = replay[:-1]
        for token in replay:
            self.next_logits = self.inference.forward_one(
                mx.array([[token]]), self.audio_features
            )
        self.prefix.extend(result.tokens)
        return result


class AssistantSpeculativeDecoder:
    """Single-sequence assistant decoding composed from the shared verifier."""

    def __init__(
        self,
        *,
        target_model,
        assistant_model,
        audio_features: mx.array,
        assistant_audio_features: Optional[mx.array] = None,
        prefix: Sequence[int],
        lookahead: int,
        eot: int,
        logit_filters: Optional[Sequence[object]] = None,
    ):
        from .decoding import Inference

        target_inference = Inference(target_model, initial_token_length=len(prefix))
        target_next_logits = target_inference.prefill(
            mx.array([list(prefix)]), audio_features
        )
        self.proposer = AssistantProposer(
            assistant_model,
            (
                assistant_audio_features
                if assistant_audio_features is not None
                else audio_features
            ),
            mx.array([list(prefix)]),
            lookahead,
        )
        self.verifier = CachedGreedyVerifier(
            inference=target_inference,
            audio_features=audio_features,
            prefix=prefix,
            next_logits=target_next_logits,
            eot=eot,
            logit_filters=logit_filters,
        )
        self.eot = eot

    def decode(self, sample_len: int):
        generated: List[int] = []
        sum_logprob = 0.0
        while len(generated) < sample_len and self.verifier.remaining_tokens:
            remaining = sample_len - len(generated)
            proposals = self.proposer.propose(
                max_tokens=self.verifier.proposal_limit(remaining)
            )
            result = self.verifier.verify(proposals, max_tokens=remaining)
            generated.extend(result.tokens[:remaining])
            sum_logprob += sum(result.token_logprobs[:remaining])
            if self.eot in result.tokens or len(generated) == sample_len:
                break
            self.proposer.synchronize(result.accepted, result.tokens)
        return generated[:sample_len], float(sum_logprob)


class MTPProposer:
    """Generate future-token proposals from the target decoder hidden state."""

    def __init__(
        self, *, inference, next_logits, head, output_projection, lookahead: int
    ):
        self.inference = inference
        self.next_logits = next_logits
        self.head = head
        self.output_projection = output_projection
        self.lookahead = lookahead

    def propose(self, max_tokens: Optional[int] = None) -> List[int]:
        width = (
            self.lookahead if max_tokens is None else min(self.lookahead, max_tokens)
        )
        if width <= 0:
            return []
        first = int(mx.argmax(self.next_logits[:, -1], axis=-1)[0].item())
        proposals = [first]
        if width == 1:
            return proposals
        if self.inference.last_hidden is None:
            raise RuntimeError("MTP proposal requires decoder hidden states")
        future_logits = self.head(
            self.inference.last_hidden[:, -1:], self.output_projection
        )[0, 0]
        for logits in future_logits[: width - 1]:
            proposals.append(int(mx.argmax(logits, axis=-1).item()))
        return proposals


class MTPSpeculativeDecoder:
    """MTP decoding composed from target hidden states and shared verification."""

    def __init__(
        self,
        *,
        model,
        head,
        audio_features: mx.array,
        prefix: Sequence[int],
        lookahead: int,
        eot: int,
        logit_filters: Optional[Sequence[object]] = None,
    ):
        from .decoding import Inference

        inference = Inference(model, initial_token_length=len(prefix))
        next_logits = inference.prefill(mx.array([list(prefix)]), audio_features)
        self.proposer = MTPProposer(
            inference=inference,
            next_logits=next_logits,
            head=head,
            output_projection=model.decoder.token_embedding.as_linear,
            lookahead=lookahead,
        )
        self.verifier = CachedGreedyVerifier(
            inference=inference,
            audio_features=audio_features,
            prefix=prefix,
            next_logits=next_logits,
            eot=eot,
            logit_filters=logit_filters,
        )
        self.eot = eot

    def decode(self, sample_len: int):
        generated: List[int] = []
        sum_logprob = 0.0
        while len(generated) < sample_len and self.verifier.remaining_tokens:
            remaining = sample_len - len(generated)
            self.proposer.next_logits = _filtered_row(
                self.verifier.next_logits[:, -1],
                self.verifier.prefix,
                self.verifier.logit_filters,
            )[:, None, :]
            proposals = self.proposer.propose(
                max_tokens=self.verifier.proposal_limit(remaining)
            )
            result = self.verifier.verify(proposals, max_tokens=remaining)
            generated.extend(result.tokens[:remaining])
            sum_logprob += sum(result.token_logprobs[:remaining])
            if self.eot in result.tokens:
                break
        return generated[:sample_len], float(sum_logprob)


def validate_speculative_options(options: SpeculativeOptions, *, temperature: float):
    if options.speculative_mode not in {"none", "assistant", "mtp"}:
        raise ValueError("speculative_mode must be one of: none, assistant, mtp")
    if options.speculative_mode == "none":
        return
    if temperature != 0:
        raise ValueError("speculative decoding requires temperature=0")
    if options.lookahead < 1:
        raise ValueError("lookahead must be at least 1")
    if options.speculative_batch_policy != "lockstep":
        raise ValueError("speculative_batch_policy must be 'lockstep'")
    if options.speculative_mode == "assistant" and options.assistant_model is None:
        raise ValueError("assistant mode requires assistant_model")
    if options.speculative_mode == "mtp" and not options.mtp_checkpoint:
        raise ValueError("mtp mode requires mtp_checkpoint")


def _as_row(logits: mx.array) -> mx.array:
    return logits[None, :] if logits.ndim == 1 else logits


def _filtered_row(logits, history, filters):
    row = _as_row(logits).astype(mx.float32)
    token_history = mx.array([history])
    for logit_filter in filters:
        row = logit_filter.apply(row, token_history)
    return row


def verify_greedy_proposals(
    *,
    prefix: Sequence[int],
    proposals: Sequence[int],
    target_logits: Iterable[mx.array],
    eot: int,
    logit_filters: Optional[Sequence[object]] = None,
) -> VerificationResult:
    """Verify proposals against filtered target argmax logits.

    ``target_logits[i]`` predicts token ``i`` from the accepted prefix plus all
    earlier proposals. The optional final row supplies the target bonus token
    after a fully accepted proposal block.
    """
    logits = list(target_logits)
    if len(logits) < len(proposals):
        raise ValueError("target_logits must cover every proposal")

    history = list(prefix)
    generated: List[int] = []
    accepted = 0
    rejected = 0
    sum_logprob = 0.0
    token_logprobs: List[float] = []
    filters = tuple(logit_filters or ())

    for index, proposal in enumerate(proposals):
        filtered = _filtered_row(logits[index], history, filters)
        target = int(mx.argmax(filtered, axis=-1)[0].item())
        logprobs = filtered - mx.logsumexp(filtered, axis=-1, keepdims=True)
        selected = target
        if proposal == target:
            accepted += 1
            selected = proposal
        else:
            rejected += 1
        generated.append(selected)
        history.append(selected)
        token_logprob = float(logprobs[0, selected].item())
        token_logprobs.append(token_logprob)
        sum_logprob += token_logprob
        if selected == eot:
            break
        if proposal != target:
            break

    bonus_tokens = 0
    if (
        accepted == len(proposals)
        and len(logits) > len(proposals)
        and (not generated or generated[-1] != eot)
    ):
        filtered = _filtered_row(logits[len(proposals)], history, filters)
        bonus = int(mx.argmax(filtered, axis=-1)[0].item())
        generated.append(bonus)
        bonus_tokens = 1
        logprobs = filtered - mx.logsumexp(filtered, axis=-1, keepdims=True)
        token_logprob = float(logprobs[0, bonus].item())
        token_logprobs.append(token_logprob)
        sum_logprob += token_logprob

    return VerificationResult(
        tokens=generated,
        accepted=accepted,
        rejected=rejected,
        bonus_tokens=bonus_tokens,
        token_logprobs=token_logprobs,
        sum_logprob=sum_logprob,
    )
