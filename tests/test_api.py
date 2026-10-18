import inspect
import pytest

from lightning_whisper_mlx import LightningWhisperMLX


def test_lightning_api_exposes_speculative_configuration():
    parameters = inspect.signature(LightningWhisperMLX.__init__).parameters

    assert parameters["speculative_mode"].default == "none"
    assert parameters["lookahead"].default == 2
    assert parameters["assistant_model"].default is None
    assert parameters["mtp_checkpoint"].default is None
    assert parameters["speculative_batch_policy"].default == "lockstep"


@pytest.mark.parametrize("quant", [None, "4bit", "8bit"])
def test_assistant_alias_resolves_for_all_supported_quantization(quant, monkeypatch):
    from lightning_whisper_mlx import lightning

    model = LightningWhisperMLX(
        model="base",
        speculative_mode="assistant",
        assistant_model="tiny",
        assistant_quant=quant,
        batch_size=1,
    )
    assert (
        model.speculative_options.assistant_model
        == lightning.model_spec("tiny", quant).repository
    )


@pytest.mark.parametrize("mode", ["assistant", "mtp"])
def test_speculative_batch_rejected_before_download(mode, monkeypatch):
    from lightning_whisper_mlx import lightning

    def unexpected_download(**kwargs):
        pytest.fail("invalid configuration must not start a download")

    monkeypatch.setattr(lightning, "resolve_model_path", unexpected_download)
    with pytest.raises(ValueError, match="batch_size=1"):
        LightningWhisperMLX(
            model="base",
            speculative_mode=mode,
            assistant_model="tiny" if mode == "assistant" else None,
            mtp_checkpoint="head" if mode == "mtp" else None,
            batch_size=2,
        )


def test_transcription_rejects_assistant_with_different_mel_bins(monkeypatch):
    from types import SimpleNamespace
    import mlx.core as mx
    from lightning_whisper_mlx import transcribe

    def model(n_mels):
        return SimpleNamespace(
            dims=SimpleNamespace(n_mels=n_mels, n_vocab=51865, n_text_ctx=16),
            is_multilingual=True,
            num_languages=99,
        )

    monkeypatch.setattr(transcribe.ModelHolder, "get_model", lambda *args: model(80))
    monkeypatch.setattr(transcribe, "load_model", lambda *args, **kwargs: model(128))
    monkeypatch.setattr(
        transcribe,
        "log_mel_spectrogram_from_prepared",
        lambda *args, **kwargs: mx.zeros((3000, 80)),
    )
    with pytest.raises(ValueError, match="mel bin count"):
        transcribe.transcribe_audio(
            mx.array([0.0, 1.0]),
            batch_size=1,
            language="en",
            speculative_mode="assistant",
            assistant_model="local-assistant",
        )


def test_distil_assistant_is_deferred_without_constructor_download(monkeypatch):
    from lightning_whisper_mlx import lightning

    monkeypatch.setattr(
        lightning,
        "resolve_model_path",
        lambda *args, **kwargs: pytest.fail("constructor must not download"),
    )
    model = LightningWhisperMLX(
        "base",
        speculative_mode="assistant",
        assistant_model="distil-small.en",
        batch_size=1,
    )
    assert model.speculative_options.assistant_model == "distil-small.en"


@pytest.mark.parametrize("checkpoint_options", [{}, {"mtp_checkpoint": None}])
def test_transcription_reports_missing_mtp_checkpoint(monkeypatch, checkpoint_options):
    from types import SimpleNamespace
    import mlx.core as mx
    from lightning_whisper_mlx import transcribe

    target = SimpleNamespace(
        dims=SimpleNamespace(n_mels=80), is_multilingual=True, num_languages=99
    )
    monkeypatch.setattr(transcribe.ModelHolder, "get_model", lambda *args: target)
    monkeypatch.setattr(
        transcribe,
        "log_mel_spectrogram_from_prepared",
        lambda *args, **kwargs: mx.zeros((3000, 80)),
    )
    with pytest.raises(ValueError, match="requires mtp_checkpoint"):
        transcribe.transcribe_audio(
            mx.array([0.0, 1.0]),
            batch_size=1,
            language="en",
            speculative_mode="mtp",
            **checkpoint_options,
        )
