import mlx.core as mx

from lightning_whisper_mlx.decoding import Inference
from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.speculative import MTPProposer
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def test_mtp_proposer_uses_target_hidden_state():
    model = Whisper(
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
    inference = Inference(model, initial_token_length=2)
    audio = mx.random.normal((1, 3, 8))
    next_logits = inference.prefill(mx.array([[1, 2]]), audio)
    proposer = MTPProposer(
        inference=inference,
        next_logits=next_logits,
        head=MTPHead(hidden_size=8, horizons=1),
        embedding_weight=model.decoder.token_embedding.weight,
        lookahead=2,
    )

    assert len(proposer.propose()) == 2
