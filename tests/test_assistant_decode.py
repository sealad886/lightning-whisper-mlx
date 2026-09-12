import mlx.core as mx

from lightning_whisper_mlx.speculative import AssistantSpeculativeDecoder
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


def test_assistant_decoder_returns_bounded_verified_sequence():
    model = make_model()
    decoder = AssistantSpeculativeDecoder(
        target_model=model,
        assistant_model=model,
        audio_features=mx.random.normal((1, 3, 8)),
        prefix=[1, 2],
        lookahead=2,
        eot=31,
    )

    tokens, sum_logprob = decoder.decode(sample_len=5)

    assert len(tokens) <= 5
    assert isinstance(sum_logprob, float)
