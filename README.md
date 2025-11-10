# Divergence From Reference (DiFR)

This is the codebase for the paper TODO.

# Setup

```
uv sync
source .venv/bin/activate
```

# Usage

To generate Token-DiFR scores, run `python difr/token_difr_vllm.py`. This is a simple, single file implementation which only calculates Token-DiFR and token probability.

To generate Token-DiFR, TOPLOC, and Activation-DiFR scores, run `python difr/vllm_verification.py`. This file isn't as optimized and supports multiple metrics. It also supports doing generation with HuggingFace, and generating with a simulated bug.

To generate plots, run `python difr/plotting.py`.