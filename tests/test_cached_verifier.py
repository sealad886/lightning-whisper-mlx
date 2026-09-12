import mlx.core as mx

from lightning_whisper_mlx.decoding import Inference
from lightning_whisper_mlx.speculative import CachedGreedyVerifier
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def make_model():
    return Whisper(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=3,
            n_audio_state=8,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=32,
            n_text_ctx=16,
            n_text_state=8,
            n_text_head=2,
            n_text_layer=1,
        ),
        dtype=mx.float32,
    )


def test_cached_verifier_commits_only_verified_tokens():
    model = make_model()
    audio = mx.random.normal((1, 3, 8))
    prefix = [1, 2]
    inference = Inference(model, initial_token_length=len(prefix))
    next_logits = inference.prefill(mx.array([prefix]), audio)
    verifier = CachedGreedyVerifier(
        inference=inference,
        audio_features=audio,
        prefix=prefix,
        next_logits=next_logits,
        eot=31,
    )

    result = verifier.verify([3, 4])

    assert verifier.cache_length == len(prefix) + len(result.tokens)
    assert verifier.prefix[-len(result.tokens) :] == result.tokens
