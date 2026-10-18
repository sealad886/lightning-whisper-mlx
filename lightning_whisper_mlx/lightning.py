"""Resident Whisper inference sessions with explicit artifact ownership."""

import threading
from pathlib import Path

import mlx.core as mx

from .load_models import load_model, model_artifact_identity, resolve_model_path
from .model_catalog import MODEL_CATALOG, model_spec
from .speculative import SpeculativeOptions, validate_speculative_options
from .transcribe import transcribe_audio, transcribe_many_audio


class LightningWhisperMLX:
    def __init__(
        self,
        model,
        batch_size=12,
        quant=None,
        speculative_mode="none",
        lookahead=2,
        assistant_model=None,
        assistant_quant=None,
        mtp_checkpoint=None,
        speculative_batch_policy="lockstep",
        *,
        local_files_only=False,
        cache_dir=None,
        dtype=mx.float16,
        compile_encoder=True,
        compile_decoder=True,
    ):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if dtype not in (mx.float16, mx.float32):
            raise ValueError("dtype must be mlx.float16 or mlx.float32")
        self._spec = (
            None if Path(model).expanduser().exists() else model_spec(model, quant)
        )
        if assistant_model is not None:
            assistant_spec = model_spec(assistant_model, assistant_quant)
            if assistant_spec and not assistant_spec.subdirectory:
                assistant_model = assistant_spec.repository
        self.speculative_options = SpeculativeOptions(
            speculative_mode=speculative_mode,
            lookahead=lookahead,
            assistant_model=assistant_model,
            assistant_quant=assistant_quant,
            mtp_checkpoint=mtp_checkpoint,
            speculative_batch_policy=speculative_batch_policy,
        )
        validate_speculative_options(self.speculative_options, temperature=0.0)
        if speculative_mode != "none" and batch_size != 1:
            raise ValueError("speculative decoding requires batch_size=1")
        self.name = str(model)
        self.batch_size = batch_size
        self.quant = quant
        self.dtype = dtype
        self.compile_encoder = compile_encoder
        self.compile_decoder = compile_decoder
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._source = model
        self._path = None
        self._model = None
        self._identity = None
        self._assistant_path = None
        self._assistant_identity = None
        self._assistant = None
        # One session serializes its mutable execution state. Separate sessions
        # own separate models and never evict one another's resident weights.
        self._lock = threading.RLock()

    @property
    def model_path(self):
        with self._lock:
            if self._path is None:
                self._path = resolve_model_path(
                    self._source,
                    quant=self.quant,
                    local_files_only=self.local_files_only,
                    cache_dir=self.cache_dir,
                )
            return self._path

    def _get_model(self, dtype):
        identity = (model_artifact_identity(self.model_path), dtype)
        if self._model is None or identity != self._identity:
            self._model = load_model(
                str(self.model_path), dtype=dtype, model_name=self.name
            )
            if self.compile_encoder or self.compile_decoder:
                self._model.configure_execution(
                    compile_encoder=self.compile_encoder,
                    compile_decoder=self.compile_decoder,
                    joint_decoding=self.compile_decoder,
                    decode_chunk_size=4,
                )
            self._identity = identity
        return self._model

    @property
    def model(self):
        """The session's lazily loaded resident model at its default precision."""
        with self._lock:
            return self._get_model(self.dtype)

    def transcribe(self, audio_path, language=None, **decode_options):
        return self._transcribe(audio_path, language, decode_options, many=False)

    def transcribe_many(self, audio_paths, language=None, **decode_options):
        """Batch independent recordings with isolated prompts and result ordering."""
        return self._transcribe(audio_paths, language, decode_options, many=True)

    def _transcribe(self, audio, language, decode_options, *, many):
        options = {
            "speculative_mode": self.speculative_options.speculative_mode,
            "lookahead": self.speculative_options.lookahead,
            "assistant_model": self.speculative_options.assistant_model,
            "assistant_quant": self.speculative_options.assistant_quant,
            "mtp_checkpoint": self.speculative_options.mtp_checkpoint,
            "speculative_batch_policy": self.speculative_options.speculative_batch_policy,
            **decode_options,
        }
        options.setdefault("fp16", self.dtype == mx.float16)
        if (
            options.get("task", "transcribe") == "translate"
            and self._spec
            and not self._spec.translate
        ):
            raise ValueError(f"model {self.name!r} does not support translation")
        with self._lock:
            model = self._get_model(mx.float16 if options["fp16"] else mx.float32)
            if options.get("task", "transcribe") == "translate" and not getattr(
                model, "supports_translation", True
            ):
                raise ValueError(f"model {self.name!r} does not support translation")
            assistant = options.get("assistant_model")
            if options["speculative_mode"] == "assistant" and isinstance(
                assistant, (str, Path)
            ):
                assistant_quant = options.get("assistant_quant")
                # A resolved repository already selects its quantized variant.
                quant = assistant_quant if str(assistant) in MODEL_CATALOG else None
                key = (str(assistant), quant)
                if self._assistant_path is None or self._assistant_path[0] != key:
                    path = resolve_model_path(
                        assistant,
                        quant=quant,
                        local_files_only=self.local_files_only,
                        cache_dir=self.cache_dir,
                    )
                    self._assistant_path = (key, path)
                path = self._assistant_path[1]
                identity = (model_artifact_identity(path), options["fp16"])
                if self._assistant is None or identity != self._assistant_identity:
                    self._assistant = load_model(
                        str(path),
                        dtype=mx.float16 if options["fp16"] else mx.float32,
                        model_name=str(assistant),
                    )
                    self._assistant_identity = identity
                options["assistant_model"] = self._assistant
            transcribe = transcribe_many_audio if many else transcribe_audio
            return transcribe(
                audio,
                path_or_hf_repo=str(self.model_path),
                model_instance=model,
                language=language,
                batch_size=self.batch_size,
                **options,
            )
