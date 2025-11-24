#!/usr/bin/env python3
"""
Create simple bar charts over all token DIFR result files.

For each metric (margin, prob, exact_match) this script:
  - Loads every pickle in RESULTS_DIR
  - Computes the mean token-level score per file
  - Draws a single bar chart with sane, auto-chosen y-limits

Compared to earlier plotting scripts, this keeps all plotting logic in a
single function and relies on standard x-axis tick labels instead of
manually positioning text below/above the axis.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration constants
# ============================================================================
RESULTS_DIR = "token_difr_results"
TRUSTED_FILENAME = f"{RESULTS_DIR}/verification_meta-llama_Llama-3_1-8B-Instruct_vllm_bf16.pkl"

# Global variables to store percentile thresholds
MARGIN_PERCENTILE_99_9: float | None = None
PROB_PERCENTILE_99_9: float | None = None


@dataclass
class SimpleTokenMetrics:
    exact_match: bool
    prob: float
    margin: float


def get_pretty_name(file_name: str) -> str:
    """Convert file name to a prettier display name for legends and labels."""
    name = file_name.replace(".pkl", "")

    # Handle OpenRouter files with providers
    lower_name = name.lower()
    if "openrouter" in lower_name:
        if "siliconflow" in lower_name:
            return "SiliconFlow"
        if "hyperbolic" in lower_name:
            return "Hyperbolic"
        if "deepinfra" in lower_name:
            return "DeepInfra"
        if "cerebras" in lower_name:
            return "Cerebras"
        if "groq" in lower_name:
            return "Groq"
        return "OpenRouter"

    # Handle vLLM quantization types
    if "_vllm_4bit" in name:
        return "4-bit"
    if "_vllm_bf16" in name:
        return "BF16"
    if "_vllm_fp8_kv" in name:
        return "FP8 KV Cache"
    if "_vllm_fp8" in name:
        return "FP8"

    # Fallback: basic cleanup
    return name.replace("verification_", "").replace("_", " ").title()


def calculate_percentile_thresholds(trusted_file: str) -> Tuple[float, float]:
    """Calculate 99.9th percentile thresholds from trusted file for margin and prob."""
    global MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9

    if MARGIN_PERCENTILE_99_9 is not None and PROB_PERCENTILE_99_9 is not None:
        return MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9

    print(f"Loading trusted file to calculate percentiles: {trusted_file}")
    with open(trusted_file, "rb") as f:
        trusted_data = pickle.load(f)

    if "scores" not in trusted_data:
        raise ValueError(f"Trusted file does not contain 'scores' key. Available keys: {list(trusted_data.keys())}")

    all_margins: List[float] = []
    all_probs: List[float] = []

    for sequence_metrics in trusted_data["scores"]:
        for token_metric in sequence_metrics:
            margin = token_metric.margin
            prob = token_metric.prob

            if not (math.isinf(margin) or math.isnan(margin)):
                all_margins.append(margin)
            if not (math.isinf(prob) or math.isnan(prob)):
                all_probs.append(prob)

    if not all_margins:
        raise ValueError("No valid margin values found in trusted file")
    if not all_probs:
        raise ValueError("No valid prob values found in trusted file")

    MARGIN_PERCENTILE_99_9 = float(np.percentile(all_margins, 99.9))
    PROB_PERCENTILE_99_9 = float(np.percentile(all_probs, 99.9))

    print(f"Margin 99.9th percentile: {MARGIN_PERCENTILE_99_9:.6f}")
    print(f"Prob 99.9th percentile: {PROB_PERCENTILE_99_9:.6f}")

    return MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9


def extract_all_scores(data: Dict, metric: str) -> List[float]:
    """Extract all token-level scores from loaded pickle data."""
    if "scores" not in data:
        raise ValueError(f"Data does not contain 'scores' key. Available keys: {list(data.keys())}")

    all_scores: List[float] = []

    for sequence_metrics in data["scores"]:
        for token_metric in sequence_metrics:
            if metric == "margin":
                if MARGIN_PERCENTILE_99_9 is None:
                    raise RuntimeError("MARGIN_PERCENTILE_99_9 not initialized. Call calculate_percentile_thresholds().")
                score = token_metric.margin
                if math.isinf(score) or math.isnan(score) or score > MARGIN_PERCENTILE_99_9:
                    score = MARGIN_PERCENTILE_99_9
                all_scores.append(float(score))
            elif metric == "prob":
                if PROB_PERCENTILE_99_9 is None:
                    raise RuntimeError("PROB_PERCENTILE_99_9 not initialized. Call calculate_percentile_thresholds().")
                score = token_metric.prob
                if math.isinf(score) or math.isnan(score):
                    score = 0.0
                elif score > PROB_PERCENTILE_99_9:
                    score = PROB_PERCENTILE_99_9
                score = -np.log(score + 1e-8)
                all_scores.append(float(score))
            elif metric == "exact_match":
                all_scores.append(1.0 if token_metric.exact_match else 0.0)
            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from 'margin', 'prob', 'exact_match'.")

    return all_scores


def compute_file_stats(all_data: Dict[str, Dict], metric: str) -> Tuple[List[str], List[str], List[float]]:
    """Compute mean scores for each file in all_data for a given metric."""
    file_keys: List[str] = []
    pretty_names: List[str] = []
    means: List[float] = []

    for pickle_path, data in all_data.items():
        file_stem = Path(pickle_path).stem
        pretty_name = get_pretty_name(file_stem)
        try:
            scores = extract_all_scores(data, metric)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {file_stem} for metric '{metric}': {exc}")
            continue

        if not scores:
            continue

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        n_tokens = len(scores)

        file_keys.append(file_stem)
        pretty_names.append(pretty_name)
        means.append(mean_score)

        print(f"{pretty_name}: Tokens: {n_tokens}, Mean: {mean_score:.6f}, Std: {std_score:.6f}")

    return file_keys, pretty_names, means


def _metric_axis_label_and_title(metric: str) -> Tuple[str, str]:
    if metric == "margin":
        return "Token-DIFR Score", "Mean Token-DIFR Score by Model"
    if metric == "prob":
        return "Cross Entropy", "Mean Cross Entropy by Model"
    if metric == "exact_match":
        return "Exact Match Rate", "Mean Exact Match Rate by Model"
    label = f"{metric.capitalize()} Score"
    return label, f"Mean {label} by Model"


def _compute_ylim(means: List[float], metric: str) -> Tuple[float, float]:
    """Choose a sensible y-axis range for the given metric."""
    y_min = float(min(means))
    y_max = float(max(means))

    if metric == "exact_match":
        # Exact match is a probability in [0,1] and usually close to 1.
        span = y_max - y_min
        if span == 0.0:
            padding = max(0.005, 0.05 * max(abs(y_max), 1.0))
        else:
            padding = max(0.005, 0.2 * span)
        bottom = max(0.0, y_min - padding)
        top = min(1.0, y_max + padding)
        if top <= bottom:
            top = bottom + 0.01
        return bottom, top

    # For margin/prob and other non-negative metrics, anchor at zero.
    if y_max <= 0:
        return 0.0, 1.0

    padding = 0.1 * y_max
    bottom = 0.0
    top = y_max + padding
    if top <= bottom:
        top = bottom + 1.0
    return bottom, top


def plot_metric_bar_chart(pretty_names: List[str], means: List[float], metric: str, output_file: str) -> None:
    """Plot a single bar chart for a metric using standard x-axis labels."""
    if not pretty_names or not means:
        print(f"No data available for metric '{metric}', skipping plot.")
        return

    positions = np.arange(len(pretty_names))

    fig, ax = plt.subplots(figsize=(max(8.0, len(pretty_names) * 0.7), 6.0))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(1, len(pretty_names) - 1)) for i in range(len(pretty_names))]

    bars = ax.bar(positions, means, color=colors, alpha=0.8)

    y_bottom, y_top = _compute_ylim(means, metric)
    ax.set_ylim(y_bottom, y_top)

    # Numeric labels above bars
    y_span = y_top - y_bottom
    label_offset = 0.01 * y_span if y_span > 0 else 0.0
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + label_offset,
            f"{mean_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ylabel, title = _metric_axis_label_and_title(metric)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)

    ax.set_xticks(positions)
    ax.set_xticklabels(pretty_names, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved bar chart for '{metric}' to: {output_file}")


def main() -> None:
    """Main entry point for generating token DIFR bar charts."""
    # Initialize percentile thresholds once
    calculate_percentile_thresholds(TRUSTED_FILENAME)

    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"Error: Directory not found: {RESULTS_DIR}")
        return

    pickle_files = sorted(results_path.glob("*.pkl"))
    if not pickle_files:
        print(f"No pickle files found in {RESULTS_DIR}")
        return

    print(f"Found {len(pickle_files)} pickle files")
    print("=" * 70)

    all_data: Dict[str, Dict] = {}
    for pickle_file in pickle_files:
        print(f"Loading {pickle_file.name}...")
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading {pickle_file}: {exc}")
            continue
        all_data[str(pickle_file)] = data

    print("=" * 70)

    for metric in ["margin", "prob", "exact_match"]:
        print(f"\nCreating bar chart for metric: {metric}")
        file_keys, pretty_names, means = compute_file_stats(all_data, metric)
        if not file_keys:
            print(f"No valid scores for metric '{metric}', skipping.")
            continue
        output_file = f"combined_bar_{metric}.png"
        plot_metric_bar_chart(pretty_names, means, metric, output_file)

    print(f"\nFinished processing {len(all_data)} files.")


if __name__ == "__main__":
    main()

