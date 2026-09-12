import mlx.core as mx

from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.speculative import MTPSpeculativeDecoder
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def test_mtp_decoder_is_bounded_and_verified():
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
    decoder = MTPSpeculativeDecoder(
        model=model,
        head=MTPHead(hidden_size=8, horizons=2),
        audio_features=mx.random.normal((1, 3, 8)),
        prefix=[1, 2],
        lookahead=3,
        eot=31,
    )

    tokens, _ = decoder.decode(sample_len=4)

    assert len(tokens) <= 4
