#!/usr/bin/env python3
"""
Data exploration script to create bar charts for all token DIFR result files.
Creates a bar chart showing mean scores for each pickle file.
"""

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration constants
# ============================================================================
RESULTS_DIR = "token_difr_results"
METRIC = "margin"
METRIC = "prob"
TRUSTED_FILENAME = "token_difr_results/verification_meta-llama_Llama-3_1-8B-Instruct_vllm_bf16.pkl"

# Global variables to store percentile thresholds
MARGIN_PERCENTILE_99_9 = None
PROB_PERCENTILE_99_9 = None


# Define SimpleTokenMetrics locally
@dataclass
class SimpleTokenMetrics:
    exact_match: bool
    prob: float
    margin: float


def calculate_percentile_thresholds(trusted_file: str):
    """Calculate 99.9th percentile thresholds from trusted file for margin and prob.

    Args:
        trusted_file: Path to the trusted pickle file

    Returns:
        Tuple of (margin_percentile_99_9, prob_percentile_99_9)
    """
    global MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9

    if MARGIN_PERCENTILE_99_9 is not None and PROB_PERCENTILE_99_9 is not None:
        return MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9

    print(f"Loading trusted file to calculate percentiles: {trusted_file}")
    with open(trusted_file, "rb") as f:
        trusted_data = pickle.load(f)

    if "scores" not in trusted_data:
        raise ValueError(f"Trusted file does not contain 'scores' key. Available keys: {list(trusted_data.keys())}")

    # Extract all margin and prob scores
    all_margins = []
    all_probs = []

    for sequence_metrics in trusted_data["scores"]:
        for token_metric in sequence_metrics:
            margin = token_metric.margin
            prob = token_metric.prob

            # Only include finite values for percentile calculation
            if not (math.isinf(margin) or math.isnan(margin)):
                all_margins.append(margin)
            if not (math.isinf(prob) or math.isnan(prob)):
                all_probs.append(prob)

    if not all_margins:
        raise ValueError("No valid margin values found in trusted file")
    if not all_probs:
        raise ValueError("No valid prob values found in trusted file")

    # Calculate 99.9th percentile
    MARGIN_PERCENTILE_99_9 = np.percentile(all_margins, 99.9)
    PROB_PERCENTILE_99_9 = np.percentile(all_probs, 99.9)

    print(f"Margin 99.9th percentile: {MARGIN_PERCENTILE_99_9:.6f}")
    print(f"Prob 99.9th percentile: {PROB_PERCENTILE_99_9:.6f}")

    return MARGIN_PERCENTILE_99_9, PROB_PERCENTILE_99_9


def extract_all_scores(data: dict, metric: str = "margin") -> list[float]:
    """Extract all token-level scores from loaded pickle data.

    Args:
        data: Loaded pickle data (should have 'scores' key)
        metric: Which metric to extract ('margin', 'prob', or 'exact_match')

    Returns:
        List of all token-level scores
    """
    if "scores" not in data:
        raise ValueError(f"Data does not contain 'scores' key. Available keys: {list(data.keys())}")

    scores = data["scores"]
    all_scores = []

    for sequence_metrics in scores:
        for token_metric in sequence_metrics:
            if metric == "margin":
                if MARGIN_PERCENTILE_99_9 is None:
                    raise RuntimeError(
                        "MARGIN_PERCENTILE_99_9 not initialized. Call calculate_percentile_thresholds() first."
                    )
                score = token_metric.margin
                # Cap infinity and large values using 99.9th percentile from trusted file
                if math.isinf(score) or math.isnan(score) or score > MARGIN_PERCENTILE_99_9:
                    score = MARGIN_PERCENTILE_99_9
                all_scores.append(score)
            elif metric == "prob":
                if PROB_PERCENTILE_99_9 is None:
                    raise RuntimeError(
                        "PROB_PERCENTILE_99_9 not initialized. Call calculate_percentile_thresholds() first."
                    )
                score = token_metric.prob
                # Handle NaN/inf and clip to 99.9th percentile from trusted file
                if math.isinf(score) or math.isnan(score):
                    score = 0.0
                elif score > PROB_PERCENTILE_99_9:
                    score = PROB_PERCENTILE_99_9

                score = -np.log(score + 1e-8)
                all_scores.append(score)
            elif metric == "exact_match":
                # Convert boolean to int
                all_scores.append(1.0 if token_metric.exact_match else 0.0)
            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from 'margin', 'prob', 'exact_match'")

    return all_scores


def create_combined_bar_plot(all_data: dict, metric: str = "margin"):
    """Create a combined bar chart showing mean scores for all pickle files.

    Args:
        all_data: Dictionary mapping file paths to loaded data
        metric: Which metric to analyze ('margin', 'prob', or 'exact_match')
    """
    if not all_data:
        print("No data to plot")
        return

    # Extract scores and calculate statistics for each file
    file_stats = {}
    file_names = []
    for pickle_file, data in all_data.items():
        file_name = Path(pickle_file).stem
        try:
            scores = extract_all_scores(data, metric)
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                n_tokens = len(scores)
                file_stats[file_name] = {"mean": mean_score, "std": std_score, "n": n_tokens}
                file_names.append(file_name)
                print(f"{file_name}: Tokens: {n_tokens}, Mean: {mean_score:.6f}, Std: {std_score:.6f}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    if not file_stats:
        print("No scores found in any files")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(max(10, len(file_names) * 1.5), 6))

    # Prepare data for bar chart
    positions = list(range(len(file_names)))
    means = [file_stats[name]["mean"] for name in file_names]
    stds = [file_stats[name]["std"] for name in file_names]

    # Get a colormap for different colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / len(file_names)) for i in range(len(file_names))]

    # Create bar chart
    bars = ax.bar(positions, means, width=0.8, color=colors, alpha=0.7)

    # Add text above each bar showing mean
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        # Position text above the bar
        text_y = height + max(means) * 0.02  # Small offset above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"μ={mean_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Create legend entries
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=file_names[i])
        for i in range(len(file_names))
    ]

    # Set labels and title
    ax.set_ylabel(f"{metric.capitalize()} Score (Mean)", fontsize=12)
    ax.set_xlabel("File Index", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in positions])
    ax.grid(True, alpha=0.3, axis="y")

    # Add legend below the plot
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=True,
    )

    # Set y-axis limits with padding for text labels
    y_max = max(means)
    y_min = min(means)
    padding = (y_max - y_min) * 0.2 if y_max != y_min else y_max * 0.2
    ax.set_ylim(min(0, y_min - padding), y_max + padding)

    # Set title
    total_tokens = sum(stats["n"] for stats in file_stats.values())
    title = f"{metric.capitalize()} Mean by Model"
    ax.set_title(title, fontsize=13, pad=15)

    # Adjust layout to make room for legend below
    plt.tight_layout(rect=(0, 0.1, 1, 1))

    # Save the plot
    output_file = f"combined_bar_{metric}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Combined bar chart saved to: {output_file}")


def main():
    """Main function - process all pickle files in results directory."""
    # Calculate percentile thresholds from trusted file first
    calculate_percentile_thresholds(TRUSTED_FILENAME)

    results_path = Path(RESULTS_DIR)

    if not results_path.exists():
        print(f"Error: Directory not found: {RESULTS_DIR}")
        return

    # Find all pickle files
    pickle_files = sorted(results_path.glob("*.pkl"))

    if not pickle_files:
        print(f"No pickle files found in {RESULTS_DIR}")
        return

    print(f"Found {len(pickle_files)} pickle files")
    print(f"Metric: {METRIC}")
    print("=" * 70)

    # Load all pickle files
    all_data = {}
    for pickle_file in pickle_files:
        print(f"Loading {pickle_file.name}...")
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
            all_data[str(pickle_file)] = data
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue

    print("=" * 70)
    print("Creating combined bar chart...")

    # Create combined bar chart
    create_combined_bar_plot(all_data, METRIC)

    print("=" * 70)
    print(f"Processed {len(all_data)} files")


if __name__ == "__main__":
    main()
