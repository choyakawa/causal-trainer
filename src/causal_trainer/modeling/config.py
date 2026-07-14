"""Architecture configuration for the training-only decoder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping

RopeStyle = Literal["gpt-j", "gpt-neox"]


def validate_splash_block_size(sequence_length: int, requested: int, name: str) -> int:
    """Return an effective TPU Splash tile after validating static constraints.

    TPU Splash requires the packed-segment query tile and the compute KV tile
    to be multiples of the TPU's 128 lanes.  All forward/backward tiles
    also have to divide their corresponding static sequence dimension. This
    runtime-independent helper is also used during CLI validation.
    """

    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, got {sequence_length!r}")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        raise ValueError(f"{name} must be a positive integer, got {requested!r}")
    effective = min(requested, sequence_length)
    if effective % 128:
        raise ValueError(f"effective {name}={effective} must be a multiple of 128")
    if sequence_length % effective:
        raise ValueError(f"sequence length {sequence_length} must be divisible by effective {name}={effective}")
    return effective


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Static decoder architecture.

    Defaults match the architecture supported by this project. Runtime and
    optimizer settings live in their respective configuration objects.
    """

    attention_bias: bool = True
    attention_dropout: float = 0.0
    eos_token_id: int | tuple[int, ...] = 151650
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 13568
    max_position_embeddings: int = 2_097_152
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 1
    pad_token_id: int = 151329
    partial_rotary_factor: float = 0.5
    rms_norm_eps: float = 1e-6
    rope_scaling: Mapping[str, Any] | None = None
    rope_style: RopeStyle = "gpt-j"
    rope_theta: float = 1_000_000_000.0
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 185600

    def __post_init__(self) -> None:
        if isinstance(self.eos_token_id, list):
            object.__setattr__(self, "eos_token_id", tuple(self.eos_token_id))

        positive_ints = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim "
                f"({self.hidden_size} != {self.num_attention_heads} * {self.head_dim})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not isinstance(self.pad_token_id, int) or isinstance(self.pad_token_id, bool):
            raise ValueError("pad_token_id must be an integer")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must index the embedding vocabulary")
        eos_token_ids = (
            self.eos_token_id
            if isinstance(self.eos_token_id, tuple)
            else (self.eos_token_id,)
        )
        if not eos_token_ids or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < self.vocab_size
            for token_id in eos_token_ids
        ):
            raise ValueError("eos_token_id must contain valid vocabulary indices")
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError("int(head_dim * partial_rotary_factor) must be positive and even")
        if self.hidden_act != "silu":
            raise ValueError("only the silu gated MLP is supported")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")
        if self.rope_style not in {"gpt-j", "gpt-neox"}:
            raise ValueError("rope_style must be 'gpt-j' or 'gpt-neox'")
        if self.rope_scaling is not None:
            raise ValueError("rope_scaling is not supported by this fixed architecture")
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("sliding-window attention is not supported")

    @property
    def rotary_dim(self) -> int:
        """Number of leading head channels rotated by RoPE."""

        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Read both source and Transformers 5.x config spellings.

        The internal and emitted representation remains the fixed source
        format.  This normalization exists only at the read boundary.
        """

        normalized = dict(values)
        # Transformers 5 gives dtype priority and treats torch_dtype as a
        # backwards-compatible fallback.
        if "dtype" in normalized:
            if normalized["dtype"] is not None:
                normalized["torch_dtype"] = normalized["dtype"]
            elif "torch_dtype" not in normalized:
                # Do not let the internal constructor's bfloat16 default turn
                # an explicitly unknown serialized dtype into a known one.
                normalized["torch_dtype"] = None

        rope_parameters = normalized.get("rope_parameters")
        rope_scaling = normalized.get("rope_scaling")
        if rope_parameters is not None and not isinstance(rope_parameters, Mapping):
            raise ValueError("rope_parameters must be a mapping")

        # Match the local v5 conversion priority: a nonempty legacy
        # rope_scaling dictionary wins, otherwise use rope_parameters.
        effective_rope: Mapping[str, Any] | None = None
        if isinstance(rope_scaling, Mapping) and rope_scaling:
            effective_rope = rope_scaling
        elif isinstance(rope_parameters, Mapping):
            effective_rope = rope_parameters
        elif isinstance(rope_scaling, Mapping):
            effective_rope = rope_scaling

        if effective_rope is not None:
            rope_type = effective_rope.get("rope_type", effective_rope.get("type", "default"))
            if rope_type != "default":
                raise ValueError(f"only unscaled default RoPE is supported, got rope_type={rope_type!r}")
            normalized["rope_theta"] = effective_rope.get(
                "rope_theta",
                normalized.get("rope_theta", 10_000.0),
            )
            normalized["partial_rotary_factor"] = effective_rope.get(
                "partial_rotary_factor",
                normalized.get("partial_rotary_factor", 0.5),
            )
            # A canonical default-RoPE dictionary is not scaling.
            normalized["rope_scaling"] = None

        accepted = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in accepted})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("model config JSON must contain an object")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if isinstance(values["eos_token_id"], tuple):
            values["eos_token_id"] = list(values["eos_token_id"])
        return values
