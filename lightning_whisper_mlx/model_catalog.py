# Copyright (c) 2022 OpenAI (alignment masks), MIT License.
"""Named Whisper checkpoints and their published alignment metadata.

Alignment masks come from OpenAI Whisper's MIT-licensed implementation:
https://github.com/openai/whisper/blob/main/whisper/__init__.py
The Hub repositories contain MLX conversions, not newly trained models.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    alignment_heads: bytes | None = None
    translate: bool = True
    subdirectory: str = ""


# OpenAI's published base85/gzip alignment masks, keyed by original checkpoint.
ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}


def _build_catalog():
    catalog = {}
    for name in ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"):
        repository_name = "large" if name == "large-v1" else name
        base = (
            "mlx-community/whisper-tiny"
            if name == "tiny"
            else f"mlx-community/whisper-{repository_name}-mlx"
        )
        catalog[name] = {"base": ModelSpec(base, ALIGNMENT_HEADS[name])}
        for quant in ("4bit", "8bit"):
            catalog[name][quant] = ModelSpec(
                f"mlx-community/whisper-{repository_name}-mlx-{quant}",
                ALIGNMENT_HEADS[name],
            )
    for name in ("tiny.en", "base.en", "small.en", "medium.en"):
        catalog[name] = {
            "base": ModelSpec(
                f"mlx-community/whisper-{name}-mlx",
                ALIGNMENT_HEADS[name],
                translate=False,
            )
        }
    catalog["large-v3-turbo"] = {
        "base": ModelSpec(
            "mlx-community/whisper-large-v3-turbo",
            ALIGNMENT_HEADS["large-v3-turbo"],
            translate=False,
        )
    }
    catalog["large"] = catalog["large-v3"]
    catalog["turbo"] = catalog["large-v3-turbo"]
    for name in (
        "distil-small.en",
        "distil-medium.en",
        "distil-large-v2",
        "distil-large-v3",
    ):
        catalog[name] = {}
        for quant in ("base", "4bit", "8bit"):
            suffix = "" if quant == "base" else f"-{quant[0]}-bit"
            catalog[name][quant] = ModelSpec(
                "mustafaaljadery/distil-whisper-mlx",
                translate=False,
                subdirectory=f"mlx_models/{name}{suffix}",
            )
    return catalog


MODEL_CATALOG = _build_catalog()


def model_spec(source, quant=None):
    """Return named/published artifact metadata without performing I/O."""
    if quant not in (None, "4bit", "8bit"):
        raise ValueError("Quantization must be `4bit` or `8bit`")
    source = str(source)
    if source in MODEL_CATALOG:
        try:
            return MODEL_CATALOG[source][quant or "base"]
        except KeyError:
            raise ValueError(
                f"quantization {quant!r} is unavailable for {source!r}"
            ) from None
    if quant:
        raise ValueError(
            "quant selects a named catalog artifact; use an explicit quantized artifact path otherwise"
        )
    for variants in MODEL_CATALOG.values():
        for spec in variants.values():
            if source == spec.repository and not spec.subdirectory:
                return spec
    return None
