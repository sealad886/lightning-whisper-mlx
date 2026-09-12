import mlx.core as mx

from lightning_whisper_mlx.speculative import AssistantProposer
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


def test_assistant_proposer_tracks_and_resynchronizes_cache():
    proposer = AssistantProposer(
        model=make_model(),
        audio_features=mx.random.normal((1, 3, 8)),
        prefix=mx.array([[1, 2]]),
        lookahead=2,
    )

    proposals = proposer.propose()
    assert len(proposals) == 2
    assert proposer.cache_length == 4

    proposer.synchronize(accepted=1, committed_tokens=[proposals[0], 7])
    assert proposer.cache_length == 4
