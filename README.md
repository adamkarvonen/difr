# Divergence From Reference (DiFR)

This is the codebase for the paper [DiFR: Inference Verification Despite Nondeterminism](https://arxiv.org/abs/2511.20621).


**Abstract** As demand for LLM inference grows, it is becoming increasingly important that providers and their customers can verify that inference processes are performed correctly, without errors or tampering. However, re-running the same inference process twice often leads to different results due to benign numerical noise, making it difficult to distinguish legitimate variation from actual problems. To address this problem, we introduce Token-DiFR (Token-Divergence-From-Reference), a method for verifying inference outputs by comparing generated tokens against predictions made by a trusted reference implementation conditioned on the same random seed. Sampling seed synchronization tightly constrains valid outputs, leaving providers minimal room to deviate from correct inference, which allows output tokens themselves to serve as auditable evidence of correctness at zero additional cost to the provider. Token-DiFR reliably identifies sampling errors, simulated bugs, and model quantization, detecting 4-bit quantization with AUC > 0.999 within 300 output tokens. For applications requiring sample-efficient forward-pass verification, we additionally introduce Activation-DiFR, a scheme that uses random orthogonal projections to compress activations into compact fingerprints for subsequent verification. Activation-DiFR detects 4-bit quantization with AUC > 0.999 using just 2 output tokens, while reducing communication overhead by 25-75% relative to existing methods. We release an open-source integration with vLLM to accelerate practical deployment of verifiable inference.


*Note* This work is developed in parallel with [Verifying LLM Inference to Prevent Model Weight Exfiltration](https://arxiv.org/abs/2511.02620), where a variant of Token-DiFR is directly applied to detecting potential steganography in LLM outputs.


## Setup

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Generating Token-DiFR Scores

**For local vLLM comparisons (recommended):** If you want to compare Token-DiFR scores for different vLLM misconfigurations (e.g., different quantizations or sampling temperatures), run:

```bash
python difr/token_difr_vllm.py
```

This is a simple, single-file implementation that calculates Token-DiFR and token probability scores. This is likely what most users will want, as it covers most use cases and is compatible with existing inference providers.

**For multiple metrics (research):** To generate Token-DiFR, TOPLOC, and Activation-DiFR scores, run:

```bash
python difr/vllm_verification.py
```

This file supports multiple metrics, HuggingFace generation, and simulated bug generation. Note that this is primarily for research purposes—other inference providers currently do not support Activation-DiFR or TOPLOC.

### Using External Prompts and Responses

To generate Token-DiFR scores for external prompts and responses (e.g., from API providers), format your data as follows:

```json
{
    "config": {
        "model": "model_name",
        "temperature": 0.0
    },
    "samples": [
        {
            "prompt_token_ids": [1, 2, 3],
            "outputs": [
                {
                    "token_ids": [4, 5, 6]
                }
            ]
        }
    ]
}
```

**Note:** The `config` field is for metadata storage only—you can include any information you like. The token format matches vLLM output formatting.

### Querying API Providers

`difr/openrouter_api.py` queries multiple API providers on OpenRouter for a given model and saves the responses to `openrouter_responses/`.

### Configuration Examples

**Adding vLLM misconfigurations:**

```python
cfgs.append(
    AttestationConfig(
        save_filename=f"verification_{model_name_str}_vllm_4bit.pkl",
        trusted_model_name=model_name,
        vllm_args={"quantization": "bitsandbytes"},
    )
)

cfgs.append(
    AttestationConfig(
        save_filename=f"verification_{model_name_str}_vllm_fp8_kv.pkl",
        trusted_model_name=model_name,
        vllm_args={"kv_cache_dtype": "fp8", "calculate_kv_scales": True},
    )
)
```

**Adding external token sources:**

```python
external_prompt_filename = (
    "openrouter_responses/openrouter_groq_meta-llama_llama-3_1-8b-instruct_token_difr_prompts_test.json"
)
external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

cfgs.append(
    AttestationConfig(
        save_filename=f"verification_{external_save_filename}.pkl",
        trusted_model_name=model_name,
        external_prompt_filename=external_prompt_filename,
    )
)
```

### Additional notes:

> **Temperature Note:** If the provider uses vLLM's sampling strategy, you can use non-zero temperature as the sampling implementation will match our default sampling procedure. If you know the provider's sampling implementation and it differs from vLLM's, you can implement a custom sampling procedure. If you are uncertain about the exact sampling implementation, you must use temperature zero.


## Plotting Results

**Bar charts (simple method, recommended for provider comparison):**

```bash
python difr/plot_token_difr_bar_plot.py
```

This generates bar charts to visually compare average Token-DiFR scores across providers and misconfigurations.

**Classification performance plots:**

```bash
python difr/plot_classification.py
```

This generates plots of classification performance (AUC at FPR < 1%) as a function of batch size and number of tokens, as shown in the paper.

## Interpreting Results

- **Low Token-DiFR scores** indicate high confidence that the provider is serving the expected model configuration.
- **High Token-DiFR scores** do not necessarily indicate poor model quality or benchmark performance, and they may simply indicate a different configuration (e.g., a different prompt template).


## Citation

If you use this work in your research, please cite:

```bibtex
@misc{karvonen2025difrinferenceverificationdespite,
      title={DiFR: Inference Verification Despite Nondeterminism},
      author={Adam Karvonen and Daniel Reuter and Roy Rinberg and Luke Marks and Adrià Garriga-Alonso and Keri Warr},
      year={2025},
      eprint={2511.20621},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.20621},
}
```
