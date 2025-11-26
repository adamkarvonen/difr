import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union
import random

import einops
import toploc
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from vllm import LLM, SamplingParams

from difr.vllm_config import AttestationConfig


class BuggyLogitsProcessor:
    def __init__(
        self,
        percent_bugged: float,
    ):
        self.percent_bugged = percent_bugged

    def __call__(
        self,
        past_tokens_ids: Union[list[int], tuple[int]],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() < self.percent_bugged:
            k = 20
            top_k_indices = torch.topk(logits, k).indices
            random_idx = random.randint(0, k - 1)
            random_token = top_k_indices[..., random_idx]
            logits = keep_one_token(logits, random_token)

        return logits


class HF_GumbelMax_LogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        generators: list[torch.Generator],
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.generators = generators

    def __call__(self, input_ids: torch.LongTensor, scores_BV: torch.FloatTensor) -> torch.FloatTensor:
        orig_dtype = scores_BV.dtype

        batch_size = scores_BV.shape[0]

        assert batch_size == len(self.generators)

        for i in range(batch_size):
            # Apply temperature
            logits_V = scores_BV[i] / max(self.temperature, 1e-8)

            # Apply top-k/top-p filtering
            logits_V = apply_top_k_top_p(logits_V[None, :], self.top_k, self.top_p).squeeze()

            # Get probabilities
            probs_V = torch.nn.functional.softmax(logits_V, dim=-1, dtype=torch.float32)

            noise_V = torch.empty_like(probs_V)
            noise_V.exponential_(generator=self.generators[i])

            noisy_logits_V = probs_V / noise_V
            token_idx = torch.argmax(noisy_logits_V)

            scores_BV[i] = keep_one_token(scores_BV[i], token_idx)

        return scores_BV.to(orig_dtype)


def exponential_to_gumbel(exponential_noise: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Convert exponential noise E ~ Exp(1) to Gumbel noise G = -log(E).

    Args:
        exponential_noise: Tensor of exponential random variables
        epsilon: Small constant to prevent log(0)

    Returns:
        Gumbel noise tensor with same shape as input
    """
    return -torch.log(exponential_noise.clamp(min=epsilon))


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    They use 2D.
    """

    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2

    # Guard: if k > vocab_size, clamp to vocab_size (select all tokens)
    V = logits.shape[1]
    k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))

    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: int | None,
    p: float | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    They use 2D.
    """

    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2
    if k is not None:
        k = torch.full((logits.shape[0],), k, device=logits.device, dtype=torch.long)
        assert k.shape[0] == logits.shape[0], "k shape must match logits batch size"
    if p is not None:
        p = torch.full((logits.shape[0],), p, device=logits.device, dtype=torch.float32)
        assert p.shape[0] == logits.shape[0], "p shape must match logits batch size"

    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """
    Keep exactly one token per row along the last dimension.

    Args:
        scores: shape (..., V) - logits/scores tensor
        tok_idx: shape (...) - must match scores.shape[:-1]

    Returns:
        shape (..., V) with all -inf except at chosen indices
    """
    # Simple rule: tok_idx shape must match all dims except last
    assert tok_idx.shape == scores.shape[:-1], (
        f"tok_idx.shape {tok_idx.shape} must match scores.shape[:-1] {scores.shape[:-1]}"
    )
    out = torch.full_like(scores, float("-inf"))

    idx = tok_idx.unsqueeze(-1)

    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)

    return out


def get_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    logits: shape [..., V]
    returns: probabilities with same shape, normalized along the last dim
    """

    assert len(logits.shape) == 2, print(f"Expected 2D logits, got shape {logits.shape}")

    if temperature > 0.0:
        x = logits / max(temperature, 1e-8)
    else:
        # greedy: pick argmax per row
        idx = torch.argmax(logits, dim=-1)
        x = keep_one_token(logits, idx)

    x = apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


# ============================================================================
# End helper functions
# ============================================================================


def compute_margin(
    logits_V: torch.Tensor,
    exponential_noise_V: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    gold_idx: torch.Tensor,
) -> float:
    # vllm scales logits by 1/temperature before top-k/top-p, so we do it here too
    # note: We do NOT scale the real logits by temperature - otherwise the noise would vary significantly
    # with temperature as well
    temp_logits_V = logits_V.clone()
    if temperature > 0.0:
        temp_logits_V = temp_logits_V / temperature

    temp_logits_V = apply_top_k_top_p(temp_logits_V[None, :], top_k, top_p).squeeze()
    neg_inf_mask = ~torch.isfinite(temp_logits_V)
    gumbel_noise = exponential_to_gumbel(exponential_noise_V.float(), epsilon=0)
    logits_V = logits_V + (gumbel_noise * temperature)

    logits_V[neg_inf_mask] = float("-inf")

    max_token = logits_V.argmax(dim=-1)

    # Difference in observed logits
    logit_diff = logits_V[max_token] - logits_V[gold_idx]

    return float(logit_diff.float().item())


def simulated_likelihood_estimator(
    logits_V: torch.Tensor,
    exponential_noise_V: torch.Tensor,
    gold_idx: torch.Tensor,
    noise_distribution: str,
    noise_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    n_samples: int,
    epsilon: float,
) -> float:
    """
    Full Monte Carlo simulation in logit space.

    Strategy (with support heuristic):
    1. Pre-filter to support on unperturbed logits (heuristic: tokens outside won't become competitive)
    2. Draw noise tensor: [n_samples, support_size]
    3. For each sample:
       - Add noise to support logits only
       - Re-apply top-k/top-p filtering on perturbed support logits
       - Add observed Gumbel noise scaled by temperature
       - Check if gold_idx wins argmax
    4. Return log(empirical win rate)

    Strategy (without support heuristic):
    1. Draw noise tensor: [n_samples, vocab_size]
    2. Add noise to all logits, filter, add Gumbel, check winner

    Works for ANY noise distribution.

    Note: We scale Gumbels by temperature, NOT logits. This keeps the noise model
    at a consistent scale regardless of temperature, and makes filtering
    temperature-independent.

    Args:
        logits_V: Verifier's logits [V]
        gumbel_noise_V: Exponential noise (E_i where G_i = -log(E_i)) [V]
        gold_idx: Index of selected token
        noise_distribution: Type of noise ("gaussian", "uniform", etc.)
        noise_scale: Scale parameter for noise (independent of temperature)
        temperature: Sampling temperature (scales Gumbel noise only)
        top_k: Top-k filtering parameter
        top_p: Top-p filtering parameter
        n_samples: Number of MC samples
        epsilon: Numerical stability constant
        use_support_heuristic: If True, pre-filter to support for speed (matches old MC)

    Returns:
        log(P(gold_idx wins | noise model))
    """
    device = logits_V.device
    dtype = logits_V.dtype

    # Filter on unperturbed logits to get initial support
    logits_filtered_V = apply_top_k_top_p(logits_V[None, :], int(top_k * 1.5), None).squeeze()

    support_mask = torch.isfinite(logits_filtered_V)
    support_indices = support_mask.nonzero(as_tuple=True)[0]

    # Check if gold is in support
    if not support_mask[gold_idx].item():
        # Gold never considered: record true zero probability (log 0 = -inf)
        return float("-inf")

    K = len(support_indices)

    # Extract support logits and Gumbels
    logits_support_K = logits_V[support_indices]
    exponential_support_K = exponential_noise_V[support_indices]

    # Find gold position in support
    gold_pos = (support_indices == gold_idx).nonzero(as_tuple=True)[0][0].item()

    # Generate noise for support only: [n_samples, K]
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    noise_samples_NK = torch.randn((n_samples, K), device=device, dtype=dtype, generator=generator) * noise_scale

    # Vectorized MC computation on support
    perturbed_logits_NK = logits_support_K.unsqueeze(0) + noise_samples_NK  # [n_samples, K]

    probs_NK = get_probs(perturbed_logits_NK[:, :], temperature, top_k, top_p).squeeze()  # [n_samples, K]

    # Add temperature-scaled Gumbel noise
    final_probs_NK = probs_NK / exponential_support_K

    # Find winners (indices in support space)
    winners_N = torch.argmax(final_probs_NK, dim=-1)  # [n_samples]

    # Count wins
    wins = (winners_N == gold_pos).sum().item()

    # Return log probability (preserve exact zero as -inf, no flooring)
    win_rate = wins / n_samples
    if wins == 0:
        return float("-inf")
    return float(math.log(win_rate))


def _effective_filter_boundary(logits_V: torch.Tensor, kept_mask: torch.Tensor) -> float:
    """
    Return the boundary as the largest excluded logit (best outside candidate).
    If nothing is excluded (no filtering active), return -inf to disable the boundary.
    """
    excluded_mask = ~kept_mask
    if excluded_mask.any():
        return logits_V[excluded_mask].max().item()  # next just outside the kept set
    else:
        return float("-inf")  # no filter => no boundary


def set_tokenizer_pad_token(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_name: str) -> AutoTokenizer:
    if not tokenizer.pad_token and "llama" in model_name.lower():
        # This is required because Llama does not have a pad token, and we want a unique pad token when collecting the response
        # Llama has three eos_tokens, and the first one is not the typical eos token
        # This is annoying, I'm not sure why they do this, but it seems to work
        tokenizer.pad_token_id = model.config.eos_token_id[0]
        assert tokenizer.pad_token_id == 128001
    elif not tokenizer.pad_token:
        raise ValueError("Tokenizer does not have a pad token, please set it")
    return tokenizer


@dataclass
class CompletionOutput:
    """
    Represents a single generated completion for a prompt.
    The structure req.outputs[0] in your example.
    """

    token_ids: list[int]


@dataclass
class VllmStyleRequestOutput:
    """
    Represents the full input and output for a single request.
    This is the 'req' object in your example.
    """

    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]


@dataclass
class ActMetrics:
    toploc_proofs: dict[int, list[bytes]]
    mean_act: float
    softmax_denominator: float
    mean_logit: float
    down_proj_vectors: dict[int, torch.Tensor]


@dataclass
class TokenMetrics:
    toploc_metrics: dict[int, tuple[int, float, float]]
    denominator_distance: float
    mean_act_distance: float
    mean_logit_distance: float
    exact_match: bool
    prob: float
    margin: float
    down_proj_distances: dict[int, float]
    rank: int
    token_index: int = -1  # 0-based index within the generated completion


def get_act_metrics(
    acts_D: torch.Tensor,
    logits_V: torch.Tensor,
    temperature: float,
    down_proj_matrices: dict[int, torch.Tensor],
) -> ActMetrics:
    logits_V = logits_V.to(dtype=torch.float32)
    mean_acts = acts_D.mean().item()
    mean_logit = logits_V.mean().item()
    temperature = max(temperature, 1e-8)
    log_z = torch.logsumexp(logits_V / temperature, dim=-1)
    softmax_denominator = float(torch.exp(log_z).float().item())

    toploc_proofs = {}
    for topk in down_proj_matrices:
        toploc_proofs[topk] = toploc.build_proofs_bytes([acts_D.cpu()], decode_batching_size=1, topk=topk)

    down_proj_vectors = {}
    for dim in down_proj_matrices:
        down_proj_vectors[dim] = (acts_D @ down_proj_matrices[dim]).squeeze().cpu()
        assert down_proj_vectors[dim].shape[0] == dim

        if down_proj_vectors[dim].isnan().any():
            raise ValueError(f"down_proj_vectors[{dim}] is nan: {down_proj_vectors[dim]}")

    act_metrics = ActMetrics(
        toploc_proofs=toploc_proofs,
        mean_act=mean_acts,
        softmax_denominator=softmax_denominator,
        mean_logit=mean_logit,
        down_proj_vectors=down_proj_vectors,
    )

    return act_metrics


def create_down_proj_matrices(
    down_proj_dims: list[int],
    acts_D: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Generates a dictionary of Johnson-Lindenstrauss random projection matrices.

    These matrices must be generated once and then shared between the provider (server)
    and the verifier (client) to ensure attestation consistency.

    Args:
        down_proj_dims: A list of target dimensions (k) for the projections.
        acts_D: The activations to project.

    Returns:
        A dictionary mapping each dimension k to its corresponding (k, vocab_size) projection matrix.
    """
    # Create the generator once for reproducibility.
    # The seed must be a public, agreed-upon constant.
    generator = torch.Generator(device=acts_D.device)

    down_proj_matrices = {}
    for k_dim in down_proj_dims:
        # 1. Create a matrix with entries sampled from N(0, 1)
        generator.manual_seed(42)
        proj_matrix_VD = torch.randn(
            acts_D.shape[-1],
            k_dim,
            generator=generator,
            device=acts_D.device,
            dtype=acts_D.dtype,
        )

        # 2. **Crucial Step:** Normalize the matrix for JL.
        # Scale by 1/sqrt(k) to ensure entries have variance 1/k.
        proj_matrix_VD /= torch.sqrt(torch.tensor(k_dim, dtype=acts_D.dtype))

        down_proj_matrices[k_dim] = proj_matrix_VD

    return down_proj_matrices


def _as_list(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def generate_outputs_hf(
    cfg: AttestationConfig,
    prompts: list[str],
    dtype: torch.dtype,
) -> tuple[list[VllmStyleRequestOutput], dict[tuple[int], list[ActMetrics]]]:
    assert cfg.untrusted_model_type == "hf"

    model = AutoModelForCausalLM.from_pretrained(cfg.untrusted_model_name, torch_dtype=dtype, device_map="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.untrusted_model_name)
    tokenizer.padding_side = "left"
    tokenizer = set_tokenizer_pad_token(tokenizer, model, cfg.untrusted_model_name)

    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    all_input_tokens = []
    all_output_tokens = []
    all_metrics = {}
    down_proj_matrices = None

    for i in tqdm(range(0, len(prompts), cfg.hf_batch_size), desc="Generating outputs"):
        tokenized_prompts = prompts[i : i + cfg.hf_batch_size]

        generators = []

        for i in range(len(tokenized_prompts)):
            generator = torch.Generator(device=model.device)
            generator.manual_seed(cfg.sampling_seed)
            generators.append(generator)

        tokenized_prompts = tokenizer(
            tokenized_prompts,  # type: ignore
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=cfg.max_ctx_len,
            return_tensors="pt",
        ).to(model.device)

        hf_attestation_processor = HF_GumbelMax_LogitsProcessor(
            temperature=cfg.sampling_temperature,
            top_k=cfg.top_k,
            top_p=cfg.sampling_top_p,
            generators=generators,
        )

        acts_LB1D = []
        logits_LB1V = []

        def activation_saving_hook(module, input, output):
            nonlocal acts_LB1D, logits_LB1V
            acts_LB1D.append(input[0])
            logits_LB1V.append(output)

        saved_acts_handle = model.lm_head.register_forward_hook(activation_saving_hook)

        output = model.generate(
            **tokenized_prompts,
            max_new_tokens=cfg.max_decode_tokens,
            do_sample=True,
            temperature=1.0,  # this temperature is fake, sampling is done by the attestation processor
            logits_processor=LogitsProcessorList([hf_attestation_processor]),
        )
        saved_acts_handle.remove()

        batch_input_tokens = []

        acts_LB1D = torch.stack(acts_LB1D).squeeze()
        logits_LB1V = torch.stack(logits_LB1V).squeeze()

        acts_BLD = einops.rearrange(acts_LB1D, "L B D -> B L D")
        logits_BLV = einops.rearrange(logits_LB1V, "L B V -> B L V")

        assert acts_BLD.shape[:-1] == logits_BLV.shape[:-1]

        for i in range(len(tokenized_prompts["input_ids"])):
            input_tokens = tokenized_prompts["input_ids"][i][tokenized_prompts["attention_mask"][i] == 1]
            batch_input_tokens.append(input_tokens)

        batch_output_tokens = []

        for i in range(len(output)):
            output_tokens = output[i][len(tokenized_prompts["input_ids"][0]) :]
            output_tokens = output_tokens[output_tokens != tokenizer.pad_token_id]
            batch_output_tokens.append(output_tokens)

            input_tokens = tuple(batch_input_tokens[i].tolist())
            all_metrics[input_tokens] = []

            # output tokens can be less than the batch seq len because of padding
            assert len(output_tokens) <= acts_BLD.shape[1]

            for j in range(len(output_tokens)):
                acts_D = acts_BLD[i, j]
                logits_V = logits_BLV[i, j]

                if down_proj_matrices is None:
                    down_proj_matrices = create_down_proj_matrices(cfg.down_proj_dims, acts_D)

                act_metrics = get_act_metrics(acts_D, logits_V, cfg.sampling_temperature, down_proj_matrices)

                all_metrics[input_tokens].append(act_metrics)

        all_input_tokens.extend(batch_input_tokens)
        all_output_tokens.extend(batch_output_tokens)

    vllm_style_outputs = []

    assert len(all_input_tokens) == len(all_output_tokens)

    for i in range(len(all_input_tokens)):
        vllm_style_outputs.append(
            VllmStyleRequestOutput(
                prompt_token_ids=all_input_tokens[i].tolist(),
                outputs=[CompletionOutput(token_ids=all_output_tokens[i].tolist())],
            )
        )

    return vllm_style_outputs, all_metrics


def generate_outputs_vllm(
    cfg: AttestationConfig,
    prompt_token_ids: list[list[int]],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[VllmStyleRequestOutput], dict[tuple[int], list[ActMetrics]]]:
    vllm_args = dict(cfg.vllm_args)

    requested_tensor_parallel = vllm_args.pop("tensor_parallel_size", None)
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = max(1, available_gpus)

    if requested_tensor_parallel is not None:
        if requested_tensor_parallel < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {requested_tensor_parallel}")
        if available_gpus and requested_tensor_parallel > available_gpus:
            raise ValueError(
                f"tensor_parallel_size {requested_tensor_parallel} exceeds available GPUs ({available_gpus})"
            )
        tensor_parallel_size = requested_tensor_parallel

    model = LLM(
        model=cfg.untrusted_model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=(cfg.max_ctx_len + cfg.max_decode_tokens) * 2,
        enforce_eager=True,
        dtype=dtype,
        **vllm_args,
    )

    logits_processors = []
    if "buggy" in cfg.save_filename.lower():
        bugged_logits_processor = BuggyLogitsProcessor(percent_bugged=0.01)
        logits_processors.append(bugged_logits_processor)

    # Use vLLM's native Gumbel-Max sampling by passing seed directly
    sampling_params = SamplingParams(
        temperature=cfg.sampling_temperature,
        max_tokens=cfg.max_decode_tokens + 1,
        top_k=cfg.top_k,
        top_p=cfg.sampling_top_p,
        seed=cfg.sampling_seed,  # This enables deterministic Gumbel-Max sampling
        logits_processors=logits_processors,
    )

    all_metrics = {}
    down_proj_matrices = None

    for i in range(len(prompt_token_ids)):
        all_metrics[tuple(prompt_token_ids[i])] = []

    def scoring_hook(module, input, output):
        nonlocal down_proj_matrices
        for i in range(len(input[2].seq_groups)):
            seq_id = input[2].seq_groups[i].seq_ids[0]
            prompt_tokens = tuple(input[2].seq_groups[i].seq_data[seq_id].prompt_token_ids)
            token_idx = input[2].selected_token_indices[i]
            logits_V = output[i]
            acts_D = input[1][token_idx]

            if down_proj_matrices is None:
                down_proj_matrices = create_down_proj_matrices(cfg.down_proj_dims, acts_D)

            act_metrics = get_act_metrics(acts_D, logits_V, cfg.sampling_temperature, down_proj_matrices)

            all_metrics[prompt_tokens].append(act_metrics)

    logits_processor = model.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor
    saved_acts_handle = logits_processor.register_forward_hook(scoring_hook)

    try:
        outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise e
    finally:
        saved_acts_handle.remove()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return outputs, all_metrics


@torch.inference_mode()
def verify_outputs(
    outputs: list[VllmStyleRequestOutput],
    vllm_metrics: dict[tuple[int], list[ActMetrics]],
    cfg: AttestationConfig,
    dtype: torch.dtype,
) -> list[list[TokenMetrics]]:
    """
    Verifies outputs using Gumbel-Max verification.

    Returns: list[list[TokenMetrics]]
    """

    trusted_model = LLM(
        model=cfg.trusted_model_name,
        tensor_parallel_size=1,
        max_model_len=(cfg.max_ctx_len + cfg.max_decode_tokens) * 2,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=0.7,
    )

    device_for_inputs = torch.device("cuda")

    all_token_metrics: list[list[TokenMetrics]] = []
    down_proj_matrices = None

    # Create per-request generator for verification
    generator = torch.Generator(device=device_for_inputs)
    lm_head = trusted_model.llm_engine.model_executor.driver_worker.model_runner.model.lm_head.weight.T

    # Note: We skip batching for two reasons: Because we do a single forward pass, we see minimal gains from batching
    # There's significant simplicity gains here, and it makes it easier to use vLLM as a verifier if we want
    for i in tqdm(range(0, len(outputs)), desc="Verifying outputs"):
        req = outputs[i]

        prompt_token_ids: list[int] = _as_list(req.prompt_token_ids)
        gen_ids: list[int] = _as_list(req.outputs[0].token_ids)
        gen_only: list[int] = _as_list(req.outputs[0].token_ids)
        seq_concat: list[int] = prompt_token_ids + gen_ids

        acts_LD = None

        def activation_saving_hook(module, input, output):
            nonlocal acts_LD
            acts_LD = input[1]

        logits_processor = trusted_model.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor
        saved_acts_handle = logits_processor.register_forward_hook(activation_saving_hook)

        fake_sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
        )

        _ = trusted_model.generate(prompt_token_ids=seq_concat, sampling_params=fake_sampling_params, use_tqdm=False)

        saved_acts_handle.remove()

        logits_LV = acts_LD @ lm_head

        assert acts_LD is not None
        assert acts_LD.shape[:-1] == logits_LV.shape[:-1], (
            f"acts_LD.shape: {acts_LD.shape}, logits_LV.shape: {logits_LV.shape}"
        )
        assert acts_LD.shape[-1] < logits_LV.shape[-1], (
            f"acts_LD.shape[-1]: {acts_LD.shape[-1]}, logits_LV.shape[-1]: {logits_LV.shape[-1]}"
        )

        # We only need float math for filtering and CDFs
        logits_LV = logits_LV.float()

        probs_LV = get_probs(logits_LV, cfg.verification_temperature, cfg.top_k, cfg.verification_top_p)

        prompt_len = len(prompt_token_ids)
        gen_len = len(gen_ids)

        vllm_seq_act_metrics: list[ActMetrics] = vllm_metrics[tuple(prompt_token_ids)]
        assert len(vllm_seq_act_metrics) == gen_len, (
            f"vllm_seq_act_metrics: {len(vllm_seq_act_metrics)}, gen_len: {gen_len}"
        )

        seq_token_metrics: list[TokenMetrics] = []

        generator.manual_seed(cfg.verification_seed)

        for j in range(gen_len):
            # logits row that predicted the j-th generated token
            # Note: -1 is because the final generated logits were not sampled
            pos = prompt_len + j - 1  # predicts token at position L + j
            logits_V = logits_LV[pos]
            prob_V = probs_LV[pos]

            actual_token_idx = torch.tensor(gen_only[j], device=prob_V.device)

            # Draw Gumbel noise for verification (exponential noise E ~ Exp(1))
            noise_V = torch.empty_like(prob_V)
            noise_V.exponential_(generator=generator)

            actual_id = gen_only[j]

            noisy_probs_V = prob_V.div(noise_V)
            pred_id = noisy_probs_V.argmax(dim=-1)

            rank = int((noisy_probs_V > noisy_probs_V[actual_token_idx]).sum().item())

            margin = compute_margin(
                logits_V.clone(),
                noise_V,
                temperature=cfg.verification_temperature,
                top_k=cfg.top_k,
                top_p=cfg.verification_top_p,
                gold_idx=actual_token_idx,
            )

            if isinstance(actual_id, torch.Tensor):
                actual_id = actual_id.item()

            acts_D = acts_LD[pos]

            if down_proj_matrices is None:
                down_proj_matrices = create_down_proj_matrices(cfg.down_proj_dims, acts_D)

            act_metrics = get_act_metrics(acts_D, logits_V, cfg.verification_temperature, down_proj_matrices)
            orig_toploc_proofs = vllm_seq_act_metrics[j].toploc_proofs
            orig_mean_acts = vllm_seq_act_metrics[j].mean_act
            orig_mean_logit = vllm_seq_act_metrics[j].mean_logit
            orig_softmax_denominator = vllm_seq_act_metrics[j].softmax_denominator

            denominator_distance = float(act_metrics.softmax_denominator - orig_softmax_denominator)
            mean_act_distance = float(act_metrics.mean_act - orig_mean_acts)
            mean_logit_distance = float(act_metrics.mean_logit - orig_mean_logit)

            toploc_metrics = {}
            for topk in cfg.down_proj_dims:
                topk_toploc_results = toploc.verify_proofs_bytes(
                    [acts_D.cpu()],
                    orig_toploc_proofs[topk],
                    decode_batching_size=1,
                    topk=topk,
                )
                # Convert to Python floats to avoid tensor issues
                exp_mismatches = topk_toploc_results[0].exp_mismatches
                mant_err_mean = topk_toploc_results[0].mant_err_mean
                mant_err_median = topk_toploc_results[0].mant_err_median
                if isinstance(exp_mismatches, torch.Tensor):
                    exp_mismatches = float(exp_mismatches.item())
                if isinstance(mant_err_mean, torch.Tensor):
                    mant_err_mean = float(mant_err_mean.item())
                if isinstance(mant_err_median, torch.Tensor):
                    mant_err_median = float(mant_err_median.item())
                toploc_metrics[topk] = (
                    exp_mismatches,
                    mant_err_mean,
                    mant_err_median,
                )

            down_proj_distances = {}
            for dim in cfg.down_proj_dims:
                down_proj_dist = torch.norm(
                    act_metrics.down_proj_vectors[dim] - vllm_seq_act_metrics[j].down_proj_vectors[dim],
                    p=2,
                )
                if down_proj_dist.isnan():
                    raise ValueError(f"down_proj_dist is nan: {down_proj_dist}")
                down_proj_distances[dim] = down_proj_dist.item()

            token_metrics = TokenMetrics(
                toploc_metrics=toploc_metrics,
                denominator_distance=denominator_distance,
                mean_act_distance=mean_act_distance,
                mean_logit_distance=mean_logit_distance,
                exact_match=bool(pred_id == actual_id),
                prob=float(prob_V[actual_token_idx].item()),
                margin=float(margin),
                down_proj_distances=down_proj_distances,
                token_index=j,  # 0-based index within this completion
                rank=rank,
            )

            seq_token_metrics.append(token_metrics)

        all_token_metrics.append(seq_token_metrics)

    return all_token_metrics


def construct_dataset(cfg: AttestationConfig) -> tuple[list[list[int]], list[str], list[list[dict[str, str]]]]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.trusted_model_name)
    tokenizer.padding_side = "left"

    ds = load_dataset(cfg.dataset_name, split="train")

    # prepare raw prompts (chat template -> text -> token ids)

    tokenized_prompts = []
    hf_prompts = []  # used because we need to add padding to prompts with hf
    conversation_prompts = []
    unique_prompts = set()

    system_prompt = []

    count = 0
    while len(tokenized_prompts) < cfg.n_samples:
        raw_prompt = ds[count]["conversation"]  # type: ignore[index]

        # Check language before processing
        if ds[count]["language"].lower() != "english":  # type: ignore[index]
            count += 1
            continue

        count += 1

        # Only include conversations that end with a user message
        # If it ends with assistant, remove the last assistant message(s) to ensure
        # the model has something to respond to
        conversation = list(raw_prompt)  # Convert to list to allow modification
        conversation = system_prompt + conversation
        while conversation and conversation[-1].get("role") == "assistant":
            conversation = conversation[:-1]

        if "qwen" in cfg.trusted_model_name.lower():
            conversation[-1]["content"] = conversation[-1]["content"] + "/nothink"

        # Skip if conversation is empty or doesn't end with user
        if not conversation or conversation[-1].get("role") != "user":
            continue

        rendered_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        tokenized_prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)
        if len(tokenized_prompt) <= cfg.max_ctx_len:
            if tuple(tokenized_prompt) not in unique_prompts:
                unique_prompts.add(tuple(tokenized_prompt))
                tokenized_prompts.append(tokenized_prompt)
                hf_prompts.append(rendered_prompt)
                conversation_prompts.append(conversation)

    # We haven't properly set the pad token yet so we need to delete the tokenizer
    del tokenizer

    return tokenized_prompts, hf_prompts, conversation_prompts


def run(cfg: AttestationConfig) -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(cfg.save_dir) / cfg.save_filename

    if save_path.exists():
        print(f"Skipping {save_path} because it already exists")
        return

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenized_prompts, hf_prompts, conversation_prompts = construct_dataset(cfg)

    if cfg.untrusted_model_type == "hf":
        untrusted_outputs, vllm_metrics = generate_outputs_hf(cfg, hf_prompts, dtype)
    else:
        untrusted_outputs, vllm_metrics = generate_outputs_vllm(cfg, tokenized_prompts, dtype, device)

    verifier_scores = verify_outputs(untrusted_outputs, vllm_metrics, cfg, dtype)

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "scores": verifier_scores,
                "config": asdict(cfg),
            },
            f,
        )


BATCH_SIZE_BY_MODEL = {
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen3-8B": 16,
    "Qwen/Qwen1.5-MoE-A2.7B": 16,
    "google/gemma-3-12b-it": 8,
    "google/gemma-2-9b-it": 8,
}

if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model_names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3-8B",
    ]

    for model_name in model_names:
        model_name_str = model_name.replace("/", "_").replace(".", "_")

        high_memory_gpu = False

        # Note: This only effects HuggingFace generate_outputs_hf()
        hf_batch_size = BATCH_SIZE_BY_MODEL[model_name]
        if high_memory_gpu:
            hf_batch_size *= 8

        cfgs = []

        # 1. Correct configuration (baseline)
        cfgs.append(
            AttestationConfig(
                save_filename=f"verification_{model_name_str}_vllm_bf16.pkl",
                trusted_model_name=model_name,
                untrusted_model_name=model_name,
                untrusted_model_type="vllm",
            )
        )

        # # buggy configuration
        # cfgs.append(
        #     AttestationConfig(
        #         save_filename=f"verification_{model_name_str}_vllm_bf16_buggy.pkl",
        #         trusted_model_name=model_name,
        #         untrusted_model_name=model_name,
        #         untrusted_model_type="vllm",
        #     )
        # )

        # # 2. KV cache quantization (FP8)
        cfgs.append(
            AttestationConfig(
                save_filename=f"verification_{model_name_str}_vllm_fp8_kv.pkl",
                trusted_model_name=model_name,
                untrusted_model_name=model_name,
                vllm_args={"kv_cache_dtype": "fp8", "calculate_kv_scales": True},
                untrusted_model_type="vllm",
            )
        )

        # # 3. Model quantization (FP8)
        # cfgs.append(
        #     AttestationConfig(
        #         save_filename=f"verification_{model_name_str}_vllm_fp8.pkl",
        #         trusted_model_name=model_name,
        #         untrusted_model_name=model_name,
        #         vllm_args={"quantization": "fp8"},
        #         untrusted_model_type="vllm",
        #     )
        # )

        # # # 4. Model quantization (4-bit)
        # # cfgs.append(
        # #     AttestationConfig(
        # #         save_filename=f"verification_{model_name_str}_vllm_4bit.pkl",
        # #         trusted_model_name=model_name,
        # #         untrusted_model_name=model_name,
        # #         vllm_args={"quantization": "bitsandbytes"},
        # #         untrusted_model_type="vllm",
        # #     )
        # # )

        # # # 5. Temperature deviation (1.0 → 1.1)
        # cfgs.append(
        #     AttestationConfig(
        #         save_filename=f"verification_{model_name_str}_vllm_bf16_temperature_1_1.pkl",
        #         trusted_model_name=model_name,
        #         untrusted_model_name=model_name,
        #         untrusted_model_type="vllm",
        #         sampling_temperature=1.1,
        #         verification_temperature=1.0,
        #     )
        # )

        # # cfgs.append(
        # #     AttestationConfig(
        # #         save_filename=f"verification_{model_name_str}_vllm_bf16_top_p_0_85.pkl",
        # #         trusted_model_name=model_name,
        # #         untrusted_model_name=model_name,
        # #         untrusted_model_type="vllm",
        # #         sampling_top_p=0.85,
        # #         verification_top_p=0.95,
        # #     )
        # # )

        # # 6. Sampling seed change (42 → 43)
        # cfgs.append(
        #     AttestationConfig(
        #         save_filename=f"verification_{model_name_str}_vllm_bf16_seed_43.pkl",
        #         trusted_model_name=model_name,
        #         untrusted_model_name=model_name,
        #         untrusted_model_type="vllm",
        #         sampling_seed=43,
        #     )
        # )

        # # 7. HuggingFace baseline (robustness check)
        # cfgs.append(
        #     AttestationConfig(
        #         save_filename=f"verification_{model_name_str}_hf_bf16.pkl",
        #         trusted_model_name=model_name,
        #         untrusted_model_name=model_name,
        #         untrusted_model_type="hf",
        #     )
        # )

        # Apply smoketest settings to all configs
        filenames = []
        for i in range(len(cfgs)):
            cfgs[i].hf_batch_size = hf_batch_size
            cfgs[i].n_samples = 200  # Smoketest: small sample
            cfgs[i].max_decode_tokens = 200  # Smoketest: minimal tokens
            cfgs[i].save_dir = f"{model_name_str}_results"
            filenames.append(cfgs[i].save_filename)

        # Check for duplicates
        assert len(filenames) == len(set(filenames)), f"Duplicate filenames: {filenames}"

        print(f"Running {len(cfgs)} configurations:")
        for i, cfg in enumerate(cfgs, 1):
            print(f"  {i}. {cfg.save_filename}")
        print("=" * 80)
        print()

        for i, cfg in enumerate(cfgs, 1):
            print(f"[{i}/{len(cfgs)}] Running: {cfg.save_filename}")
            run(cfg)
            print(f"✓ Complete: {cfg.save_filename}")
            print()
