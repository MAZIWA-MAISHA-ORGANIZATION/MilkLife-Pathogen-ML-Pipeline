"""Explainable AI (XAI) module for the MilkLife pathogen classifier.

Uses SHAP (SHapley Additive exPlanations) to attribute per-channel,
per-timestep importance scores to model predictions. This ensures
the 1D-CNN remains transparent and auditable — a hard requirement
for any diagnostic system subject to clinical or regulatory review.

GradientExplainer is chosen over DeepExplainer because it handles
the non-linearities introduced by separable convolutions and ReLU6
activations more robustly, and works with TensorFlow 2.x eager mode
without requiring graph-level shim layers.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow import keras

from src.config import DATA, PATHS, SENSOR
from src.data_loader import generate_dataset


# ── Path constants ───────────────────────────────────────────────────
_RESULTS_DIR = os.path.join(PATHS.base_dir, "results")
_SUMMARY_PLOT_PATH = os.path.join(_RESULTS_DIR, "shap_summary.png")
_CHANNEL_BAR_PATH = os.path.join(_RESULTS_DIR, "shap_channel_importance.png")


def _load_model(model_path: Optional[str] = None) -> keras.Model:
    """Loads the trained Keras model from disk.

    Args:
        model_path: Override path to a .keras checkpoint.

    Returns:
        Compiled Keras model.

    Raises:
        FileNotFoundError: If no checkpoint exists at the resolved path.
    """
    model_path = model_path or PATHS.best_model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    return keras.models.load_model(model_path)


def compute_shap_values(
    model: keras.Model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
) -> np.ndarray:
    """Computes SHAP values using GradientExplainer.

    GradientExplainer approximates Shapley values via expected
    gradients — an extension of integrated gradients that averages
    over a background distribution rather than a single baseline.

    Args:
        model: Trained Keras model.
        X_background: Reference samples used to estimate the expected
            model output (typically a stratified subset of training data).
        X_explain: Samples whose predictions will be explained.

    Returns:
        SHAP values array of shape (num_classes, N, seq_len, num_channels),
        where each entry quantifies how much a given timestep × channel
        pushed the prediction toward a specific class.
    """
    explainer = shap.GradientExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)
    return np.array(shap_values)


def plot_channel_importance(
    shap_values: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Generates a bar chart of mean |SHAP| importance per sensor channel.

    Aggregates across all timesteps and explained samples to produce
    a single importance score per channel per class, answering the
    question: "Which physical sensor most drives each classification?"

    Args:
        shap_values: Array of shape (num_classes, N, seq_len, num_channels).
        save_path: Output path for the PNG. Defaults to results/ directory.
    """
    save_path = save_path or _CHANNEL_BAR_PATH
    num_classes = shap_values.shape[0]
    channel_names = list(SENSOR.channel_names)

    # Mean absolute SHAP per channel, averaged over timesteps and samples
    # Result shape: (num_classes, num_channels)
    importance = np.mean(np.abs(shap_values), axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(channel_names))
    width = 0.18

    palette = sns.color_palette("viridis", num_classes)
    for cls_idx in range(num_classes):
        offset = (cls_idx - num_classes / 2 + 0.5) * width
        ax.bar(
            x + offset,
            importance[cls_idx],
            width=width,
            label=DATA.class_names[cls_idx],
            color=palette[cls_idx],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Sensor Channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(
        "Per-Channel Feature Importance by Pathogen Class",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["ISFET pH", "Conductivity (µS/cm)", "RGB Colorimetric"],
        fontsize=11,
    )
    ax.legend(title="Class", fontsize=10, title_fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    sns.despine()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Channel importance plot saved to: {save_path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Generates a SHAP summary beeswarm plot for each classification target.

    The plot flattens (timestep × channel) into feature columns so that
    the standard SHAP beeswarm can show which sensor regions most
    influence each class prediction.

    Args:
        shap_values: Array of shape (num_classes, N, seq_len, num_channels).
        X_explain: The samples being explained, shape (N, seq_len, num_channels).
        save_path: Output path for the PNG.
    """
    save_path = save_path or _SUMMARY_PLOT_PATH
    num_classes = shap_values.shape[0]
    n_samples = X_explain.shape[0]

    # Build human-readable feature names: "ch_name @ t=<step>"
    feature_names = []
    for t in range(SENSOR.sequence_length):
        for ch_name in SENSOR.channel_names:
            feature_names.append(f"{ch_name}@t{t}")

    # Flatten spatial dims → (N, seq_len * num_channels)
    X_flat = X_explain.reshape(n_samples, -1)

    fig, axes = plt.subplots(
        1, num_classes, figsize=(7 * num_classes, 8), squeeze=False
    )

    for cls_idx in range(num_classes):
        sv_flat = shap_values[cls_idx].reshape(n_samples, -1)

        # Select top-K features by mean |SHAP| for readability
        top_k = 20
        mean_abs = np.mean(np.abs(sv_flat), axis=0)
        top_indices = np.argsort(mean_abs)[-top_k:][::-1]

        plt.sca(axes[0, cls_idx])
        shap.summary_plot(
            sv_flat[:, top_indices],
            X_flat[:, top_indices],
            feature_names=[feature_names[i] for i in top_indices],
            show=False,
            plot_size=None,
            max_display=top_k,
        )
        axes[0, cls_idx].set_title(
            f"Class: {DATA.class_names[cls_idx]}",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(
        "SHAP Summary — Top-20 Features per Class",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ SHAP summary plot saved to: {save_path}")


def explain(
    model_path: Optional[str] = None,
    num_background: int = 100,
    num_explain: int = 50,
) -> np.ndarray:
    """End-to-end XAI pipeline: load model, compute SHAP, save plots.

    Args:
        model_path: Optional override for the trained model path.
        num_background: Number of training samples used as the SHAP
            reference distribution.
        num_explain: Number of test samples to explain.

    Returns:
        Computed SHAP values array.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    print("\n══════════════════════════════════════════════════════")
    print("   MilkLife Pathogen ML Pipeline — Explainability    ")
    print("══════════════════════════════════════════════════════\n")

    # ── Data ─────────────────────────────────────────────────────────
    X_train, X_test, _, y_test = generate_dataset()

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_train), size=num_background, replace=False)
    X_background = X_train[bg_idx]

    ex_idx = rng.choice(len(X_test), size=num_explain, replace=False)
    X_explain = X_test[ex_idx]

    print(f"  Background samples : {num_background}")
    print(f"  Explained samples  : {num_explain}")
    print(f"  Input shape        : {X_explain.shape}\n")

    # ── Model ────────────────────────────────────────────────────────
    model = _load_model(model_path)

    # ── SHAP ─────────────────────────────────────────────────────────
    print("  Computing SHAP values (GradientExplainer)...")
    shap_values = compute_shap_values(model, X_background, X_explain)
    print(f"  SHAP values shape  : {shap_values.shape}\n")

    # ── Visualisation ────────────────────────────────────────────────
    plot_channel_importance(shap_values)
    plot_shap_summary(shap_values, X_explain)

    print("\n  ✓ Explainability analysis complete.\n")
    return shap_values


if __name__ == "__main__":
    explain()
