import inspect

from lightning_whisper_mlx import LightningWhisperMLX


def test_lightning_api_exposes_speculative_configuration():
    parameters = inspect.signature(LightningWhisperMLX.__init__).parameters

    assert parameters["speculative_mode"].default == "none"
    assert parameters["lookahead"].default == 2
    assert parameters["assistant_model"].default is None
    assert parameters["mtp_checkpoint"].default is None
    assert parameters["speculative_batch_policy"].default == "lockstep"
