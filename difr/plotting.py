# %%
# %%

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings
import itertools
import math
from matplotlib.ticker import ScalarFormatter

warnings.filterwarnings("ignore")


@dataclass
class TokenMetrics:
    toploc_metrics: tuple[int, float, float]
    denominator_distance: float
    mean_act_distance: float
    mean_logit_distance: float
    exact_match: bool
    prob: float
    sampler_scores: dict[float, float]
    mc_scores: dict[float, float]
    down_proj_distances: dict[int, float]
    fast_gaussian_scores: dict[float, float]
    gumbel_max_score: float
    pairwise_gumbel_scores: dict[float, float]


@dataclass
class SimpleTokenMetrics:
    exact_match: bool
    prob: float
    sampler_scores: dict[float, float]
    mc_scores: dict[float, float]
    mc_gaussian_logit_perturbation_scores: dict[float, float]
    fast_gaussian_logit_perturbation_scores: dict[float, float]
    fast_gaussian_scores: dict[float, float]
    gumbel_max_score: float
    pairwise_gumbel_scores: dict[float, float]


SCORE_KEY_TO_LABEL = {
    "denominator_distance": "Softmax Denominator Distance",
    "mean_act_distance": "Mean Activation Distance",
    "mean_logit_distance": "Mean Logit Distance",
    "toploc_distance": "TopK Distance",
    "toploc_metrics": "Toploc (3D Features)",
    "down_proj_distances": "Johnson-Lindenstrauss Distance",
    "exact_match": "Exact Token Match",
    "prob": "Log Probability",
    "margin": "Token-DiFR Margin",
    "rank": "Rank",
}

FILENAME_TO_LABEL = {
    "vllm_bf16.pkl": "Trusted VLLM BF16",
    "vllm_bf16_top_p_0_90.pkl": "Incorrect Top-P (0.90)",
    "vllm_bf16_top_p_0_85.pkl": "Incorrect Top-P (0.85)",
    "vllm_bf16_seed_43.pkl": "Incorrect Seed (43)",
    "vllm_fp8.pkl": "Quantized VLLM FP8",
    "vllm_4bit.pkl": "Quantized VLLM 4bit",
    "vllm_fp8_kv.pkl": "Quantized VLLM KV Cache FP8",
    "hf_bf16.pkl": "Trusted HF BF16",
    "vllm_bf16_tp_2.pkl": "Trusted VLLM BF16 2 GPU Tensor Parallel",
    "vllm_bf16_tp_4.pkl": "Trusted VLLM BF16 4 GPU Tensor Parallel",
    "vllm_bf16_temperature_0_7.pkl": "Incorrect Temperature (0.7)",
    "vllm_bf16_temperature_1_1.pkl": "Incorrect Temperature (1.1)",
    "vllm_bf16_buggy.pkl": "Sampling Bug (Top-2 Bit Flip)",
    "vllm_bf16_buggy_idx_3.pkl": "Sampling Bug (4th Top Token)",
    "hf_bf16_stego.pkl": "Stego Attack HF BF16",
}


def get_features(
    data: dict, score_key: str, sigma: float = 0.01, down_proj_dim: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for classification from attestation data.

    Returns:
        seq_features: average over tokens per sequence (not used in this variant)
        token_features: one row per token
    """
    all_features = []
    token_features = []

    for i in range(len(data["scores"])):
        seq_features = []
        for j in range(len(data["scores"][i])):
            metrics: TokenMetrics = data["scores"][i][j]

            if score_key == "toploc_metrics":
                # Use all 3 toploc features
                t_exp, t_mean, t_median = metrics.toploc_metrics[down_proj_dim]
                feature = [t_exp, t_mean, t_median]
            elif score_key == "down_proj_distances":
                feature = [abs(metrics.down_proj_distances[down_proj_dim])]
            elif score_key == "prob":
                feature = [np.log(metrics.prob + 1e-8)]
            elif score_key == "exact_match":
                feature = [metrics.exact_match]
            elif score_key == "margin":
                score = metrics.margin
                if math.isinf(score):
                    score = 0.5
                if score > 0.5:
                    score = 0.5
                feature = [score]
            elif score_key == "rank":
                feature = [metrics.rank]
            else:
                raise ValueError(f"Unknown score key: {score_key}")

            if any(np.isnan(f) for f in feature):
                raise ValueError(f"nan feature for {score_key} at {i}, {j}")

            seq_features.append(feature)
            token_features.append(feature)

        all_features.append(seq_features)

    token_features = np.array(token_features)

    seq_features = []
    for seq in all_features:
        seq_array = np.array(seq)
        seq_mean = np.mean(seq_array, axis=0)
        seq_features.append(seq_mean)
    seq_features = np.array(seq_features)

    return seq_features, token_features


def train_and_evaluate_classifier(
    trusted_features: np.ndarray,
    comparison_features: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, LogisticRegression, StandardScaler]:
    """Train a logistic regression classifier and return ROC data."""
    y = np.concatenate([np.ones(len(trusted_features)), np.zeros(len(comparison_features))])
    X = np.vstack([trusted_features, comparison_features])

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, clf, scaler


def compute_standardized_partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    """Compute standardized partial AUC up to max_fpr.

    Standardization follows the scikit-learn convention so that:
    - A random classifier scores 0.5
    - A perfect classifier scores 1.0
    """
    if max_fpr <= 0:
        return 0.5

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    mask = fpr <= max_fpr
    fpr_clip = fpr[mask]
    tpr_clip = tpr[mask]

    if fpr_clip.size == 0 or fpr_clip[-1] < max_fpr:
        # Interpolate TPR at max_fpr and append
        idx = np.searchsorted(fpr, max_fpr, side="right")
        if idx == 0:
            tpr_at = tpr[0]
        elif idx >= len(fpr):
            tpr_at = tpr[-1]
        else:
            f0, f1 = fpr[idx - 1], fpr[idx]
            t0, t1 = tpr[idx - 1], tpr[idx]
            if f1 == f0:
                tpr_at = t0
            else:
                tpr_at = t0 + (t1 - t0) * (max_fpr - f0) / (f1 - f0)
        fpr_clip = np.append(fpr_clip, max_fpr)
        tpr_clip = np.append(tpr_clip, tpr_at)

    # Unstandardized partial AUC
    p_auc = np.trapz(tpr_clip, fpr_clip)

    # Standardize so random = 0.5 and perfect = 1.0
    chance = 0.5 * (max_fpr**2)
    denom = max_fpr - chance
    if denom <= 0:
        return 0.5
    std_pauc = 0.5 + 0.5 * (p_auc - chance) / denom
    return float(np.clip(std_pauc, 0.0, 1.0))


def get_label_from_filename(filename: str, filename_stub: str) -> str:
    """Extract and format label from filename."""
    label = filename.replace(filename_stub, "")
    if "untrusted" in label:
        if "Llama-3_2-3B-Instruct" in label:
            label = "Incorrect Model (Llama 3.2 3B)"
        elif "Qwen3-4B" in label:
            label = "Incorrect Model (Qwen3-4B)"
        else:
            raise ValueError(f"Unknown incorrect model: {label}")
    else:
        label = FILENAME_TO_LABEL[label]
    return label


def gather_untrusted_filenames(
    folder: str,
    trusted_filename: str,
    exclude_patterns: list[str] = None,
    include_patterns: list[str] = None,
) -> list[str]:
    """Collect untrusted filenames by pattern matching."""
    if exclude_patterns is None:
        exclude_patterns = []
    if include_patterns is None:
        include_patterns = []

    filenames = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            if not any(pattern in file for pattern in exclude_patterns):
                if len(include_patterns) > 0 and not any(pattern in file for pattern in include_patterns):
                    continue
                if file != trusted_filename:
                    filenames.append(file)
    return filenames


def make_token_buckets(
    X_tokens: np.ndarray,
    bucket_size: int,
    shuffle: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Group tokens into fixed-size buckets and average within each bucket.

    Drops remainder tokens that do not fill a full bucket.
    """
    if X_tokens.ndim == 1:
        X_tokens = X_tokens.reshape(-1, 1)

    n = X_tokens.shape[0]
    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        X_tokens = X_tokens[perm]

    n_buckets = n // bucket_size
    if n_buckets == 0:
        return np.empty((0, X_tokens.shape[1]))

    trimmed = X_tokens[: n_buckets * bucket_size]
    buckets = trimmed.reshape(n_buckets, bucket_size, -1).mean(axis=1)
    return buckets


def align_bucket_counts(
    A: np.ndarray, B: np.ndarray, shuffle: bool = True, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Trim two bucket arrays to equal number of buckets."""
    m = min(len(A), len(B))
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx_A = rng.permutation(len(A))[:m]
        idx_B = rng.permutation(len(B))[:m]
        return A[idx_A], B[idx_B]
    return A[:m], B[:m]


def analyze_with_classifiers_by_file(
    folder: str,
    trusted_filename: str,
    untrusted_filenames: list[str],
    filename_stub: str,
    score_keys: list[str],
    image_folder: str,
    model_type: str,
    sigma: float = 0.01,
    down_proj_dim: int = 8,
    bucket_sizes: list[int] = None,
    test_size: float = 0.8,
    random_state: int = 42,
    save_plots: bool = True,
    show_plots: bool = False,
    shuffle_buckets: bool = True,
    grid_rows: int = 2,
    grid_cols: int = 2,
    assert_four: bool = True,
    max_fpr: float = 1.00,
):
    """
    Analyze scores using token-bucket aggregation and draw a single figure with a grid of subplots.

    For each untrusted file, for each score_key:
      - Build token-level features for trusted and untrusted
      - Aggregate tokens into buckets of size k in {1,10,100}
      - Average features within a bucket to get one sample
      - Train a classifier and record standardized partial AUC @ FPR < max_fpr
      - Plot AUC @ FPR < max_fpr as a function of bucket size on a log-x axis

    By default this asserts that the number of untrusted files is at most grid_rows * grid_cols.
    """

    if bucket_sizes is None:
        bucket_sizes = [1, 10, 100]

    os.makedirs(image_folder, exist_ok=True)

    print(f"Processing {len(untrusted_filenames)} comparison files against trusted reference")
    print(f"Trusted reference: {trusted_filename}")
    print(f"Bucket sizes: {bucket_sizes}\n")

    # Load trusted reference data once
    trusted_filepath = os.path.join(folder, trusted_filename)
    print(f"Loading trusted reference: {trusted_filename}")
    with open(trusted_filepath, "rb") as f:
        trusted_data = pickle.load(f)
    print(f"  Loaded {len(trusted_data['scores'])} sequences\n")

    # Prepare consistent styles across all subplots
    score_labels_ordered = []
    for k in score_keys:
        lbl = SCORE_KEY_TO_LABEL[k]
        if k in {"down_proj_distances", "toploc_metrics", "combined_jl_sampler"}:
            lbl += f" (k={down_proj_dim})"
        score_labels_ordered.append(lbl)

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(score_labels_ordered))))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "8"]
    style_map: dict[str, tuple] = {}
    for i, lbl in enumerate(score_labels_ordered):
        style_map[lbl] = (colors[i % len(colors)], markers[i % len(markers)])

    plot_data: list[tuple[str, dict[str, list[float]]]] = []

    # Process each comparison file
    for filename in untrusted_filenames:
        filepath = os.path.join(folder, filename)
        label = get_label_from_filename(filename, filename_stub)

        print(f"\n{'=' * 60}")
        print(f"Processing File: {label}")
        print(f"{'=' * 60}")

        with open(filepath, "rb") as f:
            comparison_data = pickle.load(f)

        auc_by_key = {lbl: [] for lbl in score_labels_ordered}

        for score_key, score_label in zip(score_keys, score_labels_ordered):
            _, trusted_token_features = get_features(trusted_data, score_key, sigma, down_proj_dim)
            _, comparison_token_features = get_features(comparison_data, score_key, sigma, down_proj_dim)

            key_aucs = []
            for k in bucket_sizes:
                trusted_buckets = make_token_buckets(
                    trusted_token_features, bucket_size=k, shuffle=shuffle_buckets, random_state=random_state
                )
                comparison_buckets = make_token_buckets(
                    comparison_token_features, bucket_size=k, shuffle=shuffle_buckets, random_state=random_state + 1
                )

                trusted_buckets, comparison_buckets = align_bucket_counts(
                    trusted_buckets, comparison_buckets, shuffle=shuffle_buckets, random_state=random_state + 2
                )

                if len(trusted_buckets) == 0 or len(comparison_buckets) == 0:
                    raise ValueError(f"No buckets available for size {k} for score key {score_key}")

                fpr, tpr, _, _, _ = train_and_evaluate_classifier(
                    trusted_buckets, comparison_buckets, test_size=test_size, random_state=random_state
                )
                std_pauc = compute_standardized_partial_auc(fpr, tpr, max_fpr=max_fpr)
                key_aucs.append(std_pauc)
                print(f"  {score_label} | bucket {k}: AUC @ FPR < {int(max_fpr * 100)}% = {std_pauc:.4f}")

            auc_by_key[score_label] = key_aucs

        plot_data.append((label, auc_by_key))

    # Build a single figure with subplots
    n_files = len(plot_data)
    capacity = grid_rows * grid_cols
    if assert_four:
        assert n_files <= capacity, f"Too many untrusted files: {n_files} > capacity {capacity}."

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(5 * grid_cols, 4.2 * grid_rows),
        sharex=True,
        sharey=True,
    )
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    # Plot into subplots
    for idx, (label, auc_by_key) in enumerate(plot_data):
        ax = axes_flat[idx]
        for slabel, aucs in auc_by_key.items():
            color, marker = style_map[slabel]
            ax.plot(
                bucket_sizes,
                aucs,
                marker=marker,
                linewidth=2,
                label=slabel,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_xticks(bucket_sizes)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    # Turn off any unused subplots
    for j in range(n_files, capacity):
        axes_flat[j].axis("off")

    # Common labels
    for ax in axes_flat[: max(n_files, 1)]:
        ax.set_xlabel("Token bucket size (log scale)")
        ax.set_ylabel(f"AUC @ FPR < {int(max_fpr * 100)}%")

    # Single shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(score_labels_ordered), 3),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        f"AUC @ FPR < {int(max_fpr * 100)}% vs Token Bucket Size (sigma {sigma}) for {model_type} - All Comparisons",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))

    # Save or show
    if save_plots:
        out_path = f"{image_folder}/line_auc_vs_bucket_grid_{model_type}_k_{down_proj_dim}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Grid figure saved as {out_path}")
    if show_plots:
        plt.show()

    # Return results keyed by file label
    return {label: auc_by_key for label, auc_by_key in plot_data}


# %%


# Example usage:
if __name__ == "__main__":
    # Setup data directory
    data_dir = "meta-llama_Llama-3_1-8B-Instruct_results"
    data_dir = "token_difr_results"
    model_type = "Llama-3.1-8B"

    # data_dir = "Qwen_Qwen3-8B_results"
    # model_type = "Qwen3-8B"
    os.makedirs(data_dir, exist_ok=True)
    # model_type = "qwen"
    specific_local_path = data_dir

    # Define score keys to evaluate
    score_keys = [
        "exact_match",
        "prob",
        "margin",
    ]

    # Setup parameters
    if model_type == "Llama-3.1-8B":
        trusted_filename = "verification_meta-llama_Llama-3_1-8B-Instruct_vllm_bf16.pkl"
    elif model_type == "Qwen3-8B":
        trusted_filename = "verification_Qwen_Qwen3-8B_vllm_bf16.pkl"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    filename_stub = trusted_filename.split("vllm_bf16.pkl")[0]
    folder = specific_local_path
    image_folder = f"images_line_plots_auc_vs_bucket_{model_type}"

    print("=" * 80)
    print("RUNNING CLASSIFIER-BASED ANALYSIS WITH TOKEN BUCKET AGGREGATION - SINGLE GRID FIGURE")
    print("=" * 80)

    # Gather untrusted filenames
    untrusted_filenames = gather_untrusted_filenames(
        folder=folder,
        trusted_filename=trusted_filename,
        exclude_patterns=[],
        include_patterns=[],
    )

    # If you want to hard-enforce a 2x2 grid with at most 4 files, keep assert_four=True
    # Set assert_four=False to allow more files and adjust grid_rows/grid_cols accordingly
    results = analyze_with_classifiers_by_file(
        folder=folder,
        trusted_filename=trusted_filename,
        untrusted_filenames=untrusted_filenames,
        filename_stub=filename_stub,
        score_keys=score_keys,
        image_folder=image_folder,
        model_type=model_type,
        down_proj_dim=8,
        bucket_sizes=[1, 3, 10, 30, 100, 300, 1000],
        test_size=0.5,
        random_state=43,
        save_plots=True,
        show_plots=False,
        shuffle_buckets=True,
        grid_rows=2,
        grid_cols=3,
        assert_four=False,  # set to False and change grid_rows/grid_cols if you want a larger grid
    )

# # %%

# %%
