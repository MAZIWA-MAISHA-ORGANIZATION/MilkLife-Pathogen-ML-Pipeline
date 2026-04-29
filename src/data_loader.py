"""Data ingestion and augmentation pipeline for multi-channel sensor data.

Generates synthetic multi-channel sensor readings that mimic the
statistical signatures of real ISFET pH, conductivity, and RGB
colorimetric sensor outputs. Uses tf.data.Dataset for performant
batching and prefetching suitable for GPU training.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf

from src.config import DATA, SENSOR, TRAINING


# ── Synthetic Signal Profiles ────────────────────────────────────────
# Each class has a characteristic mean and variance per channel,
# derived from domain literature on dairy pathogen metabolic byproducts.
_CLASS_PROFILES = {
    #                       pH       Conductivity   RGB intensity
    "healthy":         {"mean": [6.70, 5.20, 0.80], "std": [0.10, 0.30, 0.05]},
    "e_coli":          {"mean": [5.80, 7.10, 0.55], "std": [0.20, 0.50, 0.08]},
    "s_aureus":        {"mean": [6.10, 6.50, 0.45], "std": [0.15, 0.45, 0.07]},
    "early_spoilage":  {"mean": [5.50, 8.00, 0.35], "std": [0.25, 0.60, 0.10]},
}


def _generate_synthetic_sample(
    class_name: str,
    seq_len: int,
    num_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesises a single multi-channel time-series reading.

    Applies a slow baseline drift plus high-frequency sensor noise to
    produce realistic waveform morphology per pathogen class.

    Args:
        class_name: One of the keys in DATA.class_names.
        seq_len: Number of timesteps per reading window.
        num_channels: Number of analog sensor channels.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (seq_len, num_channels).
    """
    profile = _CLASS_PROFILES[class_name]
    mean = np.array(profile["mean"][:num_channels])
    std = np.array(profile["std"][:num_channels])

    # Low-frequency baseline drift simulates thermal / reagent aging effects
    t = np.linspace(0, 1, seq_len).reshape(-1, 1)
    drift = 0.05 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)

    noise = rng.normal(loc=0.0, scale=std, size=(seq_len, num_channels))
    signal = mean + drift + noise
    return signal.astype(np.float32)


def generate_dataset(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates the full synthetic dataset and splits into train / test.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(seed)
    samples_per_class = DATA.num_samples // DATA.num_classes

    X_all, y_all = [], []
    for label_idx, class_name in enumerate(DATA.class_names):
        for _ in range(samples_per_class):
            sample = _generate_synthetic_sample(
                class_name, SENSOR.sequence_length, SENSOR.num_channels, rng
            )
            X_all.append(sample)
            y_all.append(label_idx)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)

    # Deterministic shuffle
    indices = rng.permutation(len(X_all))
    X_all, y_all = X_all[indices], y_all[indices]

    split = int(len(X_all) * (1 - DATA.test_split))
    return X_all[:split], X_all[split:], y_all[:split], y_all[split:]


# ── Augmentation ─────────────────────────────────────────────────────

def _add_gaussian_noise(
    x: tf.Tensor, y: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simulates hardware-level sensor variance via additive Gaussian noise.

    Noise standard deviation is sampled uniformly from the configured
    augmentation range so the model sees varying SNR conditions.
    """
    noise_std = tf.random.uniform(
        [],
        minval=DATA.augmentation_noise_range[0],
        maxval=DATA.augmentation_noise_range[1],
    )
    noise = tf.random.normal(shape=tf.shape(x), stddev=noise_std)
    return x + noise, y


def _time_shift(
    x: tf.Tensor, y: tf.Tensor, max_shift: int = 8
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Random circular shift along the time axis to improve temporal invariance."""
    shift = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)
    return x, y


def build_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Wraps NumPy arrays in an optimised tf.data.Dataset pipeline.

    Args:
        X: Feature array of shape (N, seq_len, num_channels).
        y: Label array of shape (N,).
        is_training: When True, applies augmentation, shuffling, and repeating.

    Returns:
        A batched, prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if is_training:
        ds = ds.shuffle(DATA.shuffle_buffer_size, reshuffle_each_iteration=True)
        ds = ds.map(_add_gaussian_noise, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_time_shift, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(TRAINING.batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
