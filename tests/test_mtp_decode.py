import mlx.core as mx

from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.speculative import MTPSpeculativeDecoder
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


def test_mtp_decoder_is_bounded_and_verified():
    model = make_model()
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


def test_mtp_first_proposal_follows_current_target_filters(monkeypatch):
    model = make_model()

    class HistoryFilter:
        def apply(self, logits, tokens):
            # Select a history-dependent token different from the raw winner.
            raw = int(mx.argmax(logits, axis=-1)[0].item())
            selected = (raw + tokens.shape[-1]) % logits.shape[-1]
            return mx.where(mx.arange(logits.shape[-1]) == selected, 0.0, -mx.inf)[
                None, :
            ]

    decoder = MTPSpeculativeDecoder(
        model=model,
        head=MTPHead(hidden_size=8, horizons=1),
        audio_features=mx.random.normal((1, 3, 8)),
        prefix=[1, 2],
        lookahead=1,
        eot=-1,
        logit_filters=[HistoryFilter()],
    )
    outcomes = []
    verify = decoder.verifier.verify

    def record_verification(proposals, **kwargs):
        result = verify(proposals, **kwargs)
        outcomes.append(result)
        return result

    monkeypatch.setattr(decoder.verifier, "verify", record_verification)
    tokens, _ = decoder.decode(sample_len=4)

    assert len(tokens) == 4
    assert outcomes
    assert all(result.accepted == 1 and result.rejected == 0 for result in outcomes)
