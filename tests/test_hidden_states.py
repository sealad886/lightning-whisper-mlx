import mlx.core as mx

from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def test_whisper_can_return_decoder_hidden_states():
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
    audio = mx.random.normal((1, 3, 8))
    tokens = mx.array([[1, 2]])

    logits, hidden = model.logits_and_hidden(tokens, audio)

    assert logits.shape == (1, 2, 32)
    assert hidden.shape == (1, 2, 8)
