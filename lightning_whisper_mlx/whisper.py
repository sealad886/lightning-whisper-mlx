# Copyright © 2023 Apple Inc.

import base64
import gzip
import math
from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .cache import KVCache


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.n_head = n_head
        self.cross_attention = cross_attention
        if cross_attention:
            self.query = nn.Linear(n_state, n_state)
            self.kv = nn.Linear(n_state, 2 * n_state)
            self.kv.bias = mx.concatenate(
                [mx.zeros((n_state,)), self.kv.bias[n_state:]]
            )
        else:
            self.qkv = nn.Linear(n_state, 3 * n_state)
            self.qkv.bias = mx.concatenate(
                [
                    self.qkv.bias[:n_state],
                    mx.zeros((n_state,)),
                    self.qkv.bias[2 * n_state :],
                ]
            )
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
        return_scores=False,
        cache_offset=None,
    ):
        if self.cross_attention:
            q = self.query(x)
            if kv_cache is None:
                if xa is None:
                    raise ValueError(
                        "cross attention requires audio features or a cache"
                    )
                k, v = mx.split(self.kv(xa), 2, axis=-1)
            else:
                k, v = kv_cache
        else:
            if xa is not None:
                raise ValueError(
                    "self attention does not accept cross-attention features"
                )
            q, k, v = mx.split(self.qkv(x), 3, axis=-1)
            if cache_offset is not None:
                if kv_cache is None:
                    raise ValueError("tensor cache updates require existing buffers")
                k = mx.slice_update(kv_cache[0], k, cache_offset.reshape(1), axes=(1,))
                v = mx.slice_update(kv_cache[1], v, cache_offset.reshape(1), axes=(1,))
            elif kv_cache is not None:
                k, v = kv_cache.update(k, v)

        wv, qk = self.qkv_attention(q, k, v, mask, return_scores=return_scores)
        cache = (
            kv_cache
            if not self.cross_attention
            and cache_offset is None
            and kv_cache is not None
            else (k, v)
        )
        return self.out(wv), cache, qk

    def qkv_attention(self, q, k, v, mask=None, return_scores=False):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        if not return_scores:
            if mask is not None and not isinstance(mask, str):
                mask = mask.astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            return out.transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state), None

        # Alignment requires explicit scores with the reference scaling order.
        qk = (q * scale**0.5) @ (k.swapaxes(-1, -2) * scale**0.5)
        if mask is not None:
            qk = qk + mask
        qk = qk.astype(mx.float32)

        w = mx.softmax(qk, axis=-1).astype(q.dtype)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, cross_attention=True)
            if cross_attention
            else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
        return_scores=False,
        cache_offset=None,
    ):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.attn(
            self.attn_ln(x), mask=mask, kv_cache=kv, cache_offset=cache_offset
        )
        x += y
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x),
                xa,
                kv_cache=cross_kv,
                return_scores=return_scores,
            )
            x += y
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(
        self,
        x,
        xa,
        kv_cache=None,
        return_hidden=False,
        return_scores=False,
        offset=None,
    ):
        """
        x : mx.array, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : mx.array, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        n_tokens = x.shape[-1]
        tensor_cache = offset is not None
        if tensor_cache:
            if n_tokens != 1 or kv_cache is None:
                raise ValueError(
                    "functional decoder step requires one token and existing cache buffers"
                )
            capacity = kv_cache[0][0][0].shape[1]
            if capacity > self.positional_embedding.shape[0]:
                raise ValueError("decoder cache exceeds text context")
            positions = mx.take(self.positional_embedding, offset.reshape(1), axis=0)
            x = self.token_embedding(x) + positions
            mask = mx.where(mx.arange(capacity) > offset, -float("inf"), 0).astype(
                x.dtype
            )
            # The compiled function returns state explicitly and never mutates
            # the caller's Python cache container or KVCache owners.
            kv_cache = list(kv_cache)
            cache_offset = offset
        else:
            offset = kv_cache[0][0].offset if kv_cache else 0
            if offset + n_tokens > self.positional_embedding.shape[0]:
                raise ValueError("decoder tokens exceed text context")
            x = (
                self.token_embedding(x)
                + self.positional_embedding[offset : offset + n_tokens]
            )
            if kv_cache is None:
                kv_cache = [
                    (KVCache(self.positional_embedding.shape[0]), None)
                    for _ in self.blocks
                ]
            mask = self._mask[offset : offset + n_tokens, : offset + n_tokens]
            cache_offset = None
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x,
                xa,
                mask=mask,
                kv_cache=kv_cache[e],
                return_scores=return_scores,
                cache_offset=cache_offset,
            )

        hidden = self.ln(x)
        logits = self.token_embedding.as_linear(hidden)
        if return_hidden:
            return logits, kv_cache, cross_qk, hidden
        return logits, kv_cache, cross_qk


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        if isinstance(dump, np.ndarray):
            self.alignment_heads = mx.array(dump)
        elif isinstance(dump, bytes):
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self.alignment_heads = mx.array(np.asarray(mask.nonzero()).T)
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def embed_audio(self, mel):
        encoder = getattr(self, "_compiled_encoder", None)
        return self.encoder(mel) if encoder is None else encoder(mel)

    def configure_execution(self, *, compile_encoder=False, compile_decoder=False):
        """Compile the shape-stable encoder after weights are loaded, if requested.

        Changing weights/precision requires configuring again. Generation caches
        remain request-owned and are never captured as implicit compiled state.
        """
        self._compiled_encoder = mx.compile(self.encoder) if compile_encoder else None
        decoder = self.decoder

        def decoder_step(tokens, features, state, offset):
            logits, state, _, hidden = decoder(
                tokens, features, kv_cache=state, return_hidden=True, offset=offset
            )
            return logits, state, hidden

        self._compiled_decoder = mx.compile(decoder_step) if compile_decoder else None

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def logits_and_hidden(self, tokens, audio_features):
        logits, _, _, hidden = self.decoder(tokens, audio_features, return_hidden=True)
        return logits, hidden

    def forward_with_cross_qk(self, mel, tokens, audio_features=None):
        features = self.embed_audio(mel) if audio_features is None else audio_features
        logits, _, cross_qk = self.decoder(tokens, features, return_scores=True)
        return logits, cross_qk

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    decode = decode_function
