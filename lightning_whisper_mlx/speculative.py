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
    sum_logprob: float


class AssistantProposer:
    """Greedy assistant proposal generator with cache synchronization."""

    def __init__(self, model, audio_features: mx.array, prefix: mx.array, lookahead: int):
        from .decoding import Inference

        self.inference = Inference(model, initial_token_length=prefix.shape[-1])
        self.audio_features = audio_features
        self.lookahead = lookahead
        self._next_logits = self.inference.prefill(prefix, audio_features)

    @property
    def cache_length(self) -> int:
        return self.inference.cache_length

    def propose(self) -> List[int]:
        proposals: List[int] = []
        logits = self._next_logits[:, -1]
        for _ in range(self.lookahead):
            token = int(mx.argmax(logits, axis=-1)[0].item())
            proposals.append(token)
            self._next_logits = self.inference.forward_one(
                mx.array([[token]]), self.audio_features
            )
            logits = self._next_logits[:, -1]
        return proposals

    def synchronize(self, accepted: int, committed_tokens: Sequence[int]):
        committed_prefix_length = self.cache_length - (len(committed_tokens) - accepted)
        self.inference.truncate(committed_prefix_length)
        for token in committed_tokens[accepted:]:
            self._next_logits = self.inference.forward_one(
                mx.array([[token]]), self.audio_features
            )


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

    def verify(self, proposals: Sequence[int]) -> VerificationResult:
        if not proposals:
            raise ValueError("at least one proposal is required")

        committed_length = self.inference.cache_length
        block_logits = self.inference.forward_block(
            mx.array([list(proposals)]), self.audio_features
        )
        target_logits = [self.next_logits[:, -1]]
        target_logits.extend(block_logits[0, index] for index in range(len(proposals)))
        result = verify_greedy_proposals(
            prefix=self.prefix,
            proposals=proposals,
            target_logits=target_logits,
            eot=self.eot,
            logit_filters=self.logit_filters,
        )

        self.inference.truncate(committed_length + result.accepted)
        for token in result.tokens[result.accepted :]:
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
            assistant_audio_features
            if assistant_audio_features is not None
            else audio_features,
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
        while len(generated) < sample_len:
            proposals = self.proposer.propose()
            result = self.verifier.verify(proposals)
            self.proposer.synchronize(result.accepted, result.tokens)
            generated.extend(result.tokens)
            sum_logprob += result.sum_logprob
            if self.eot in result.tokens:
                break
        return generated[:sample_len], float(sum_logprob)


class MTPProposer:
    """Generate future-token proposals from the target decoder hidden state."""

    def __init__(self, *, inference, next_logits, head, embedding_weight, lookahead: int):
        self.inference = inference
        self.next_logits = next_logits
        self.head = head
        self.embedding_weight = embedding_weight
        self.lookahead = lookahead

    def propose(self) -> List[int]:
        first = int(mx.argmax(self.next_logits[:, -1], axis=-1)[0].item())
        proposals = [first]
        if self.lookahead == 1:
            return proposals
        if self.inference.last_hidden is None:
            raise RuntimeError("MTP proposal requires decoder hidden states")
        future_logits = self.head(
            self.inference.last_hidden[:, -1:], self.embedding_weight
        )[0, 0]
        for logits in future_logits[:self.lookahead - 1]:
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
            embedding_weight=model.decoder.token_embedding.weight,
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
        while len(generated) < sample_len:
            self.proposer.next_logits = self.verifier.next_logits
            result = self.verifier.verify(self.proposer.propose())
            generated.extend(result.tokens)
            sum_logprob += result.sum_logprob
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
    filters = tuple(logit_filters or ())

    for index, proposal in enumerate(proposals):
        row = _as_row(logits[index]).astype(mx.float32)
        filtered = row
        token_history = mx.array([history])
        for logit_filter in filters:
            filtered = logit_filter.apply(filtered, token_history)
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
        sum_logprob += float(logprobs[0, selected].item())
        if selected == eot:
            break
        if proposal != target:
            break

    bonus_tokens = 0
    if accepted == len(proposals) and len(logits) > len(proposals):
        row = _as_row(logits[len(proposals)]).astype(mx.float32)
        filtered = row
        token_history = mx.array([history])
        for logit_filter in filters:
            filtered = logit_filter.apply(filtered, token_history)
        bonus = int(mx.argmax(filtered, axis=-1)[0].item())
        generated.append(bonus)
        bonus_tokens = 1
        logprobs = filtered - mx.logsumexp(filtered, axis=-1, keepdims=True)
        sum_logprob += float(logprobs[0, bonus].item())

    return VerificationResult(
        tokens=generated,
        accepted=accepted,
        rejected=rejected,
        bonus_tokens=bonus_tokens,
        sum_logprob=sum_logprob,
    )
