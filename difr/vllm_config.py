from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AttestationConfig:
    seed: int = 42
    trusted_model_name: str = "google/gemma-2-2b-it"
    untrusted_model_name: str = "google/gemma-2-2b-it"
    max_decode_tokens: int = 512
    n_samples: int = 100
    hf_batch_size: int = 10
    verification_batch_size: int | None = None
    max_ctx_len: int = 512

    untrusted_model_type: Literal["hf", "vllm"] = "vllm"

    vllm_args: dict[str, Any] = field(default_factory=dict)

    dataset_name: str = "lmsys/lmsys-chat-1m"
    dataset_split: str = "train"
    dataset_config: str | None = None
    # Sampling parameters

    sampling_temperature: float = 1.0
    verification_temperature: float = 1.0
    top_k: int = 50
    sampling_top_p: float = 0.95
    verification_top_p: float = 0.95
    sampling_seed: int = 42
    verification_seed: int = 42

    down_proj_dims: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])

    save_dir: str = "attestation_results"
    save_filename: str = "attestation_results.pkl"
