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
    head = MTPHead(hidden_size=8, horizons=1)
    observed_hidden = []

    def observe_head(hidden, output_projection):
        observed_hidden.append(hidden)
        return head(hidden, output_projection)

    proposer = MTPProposer(
        inference=inference,
        next_logits=next_logits,
        head=observe_head,
        output_projection=model.decoder.token_embedding.as_linear,
        lookahead=2,
    )

    assert len(proposer.propose()) == 2
    assert mx.array_equal(observed_hidden[-1], inference.last_hidden[:, -1:]).item()

    # Advancing committed target history must update the head's conditioning.
    proposer.next_logits = inference.forward_one(mx.array([[3]]), audio)
    assert len(proposer.propose()) == 2
    assert mx.array_equal(observed_hidden[-1], inference.last_hidden[:, -1:]).item()
