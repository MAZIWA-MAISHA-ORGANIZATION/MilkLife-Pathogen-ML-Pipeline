"""Centralized configuration for the MilkLife Pathogen ML Pipeline.

All hyperparameters, sensor channel dimensions, and path constants
are defined here to ensure reproducibility and single-source-of-truth
configuration across training, evaluation, and export stages.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass(frozen=True)
class SensorConfig:
    """Physical sensor channel dimensions for the MilkLife diagnostic probe.

    Each reading window captures `sequence_length` timesteps across
    three analog channels: ISFET pH, conductivity (µS/cm), and
    RGB colorimetric intensity (combined single channel).
    """

    num_channels: int = 3
    sequence_length: int = 128
    channel_names: tuple = ("isfet_ph", "conductivity", "rgb_colorimetric")


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters governing the training loop."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    validation_split: float = 0.15
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    dropout_rate: float = 0.3
    label_smoothing: float = 0.05


@dataclass(frozen=True)
class DataConfig:
    """Synthetic dataset generation and augmentation parameters."""

    num_samples: int = 5000
    noise_stddev: float = 0.02
    augmentation_noise_range: tuple = (0.005, 0.03)
    test_split: float = 0.20
    shuffle_buffer_size: int = 2048
    class_names: tuple = ("healthy", "e_coli", "s_aureus", "early_spoilage")
    num_classes: int = 4


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for artifacts produced by the pipeline."""

    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir: str = field(default=None)
    checkpoint_path: str = field(default=None)
    best_model_path: str = field(default=None)
    tflite_output_path: str = field(default=None)
    log_dir: str = field(default=None)

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for deferred defaults
        defaults = {
            "model_dir": os.path.join(self.base_dir, "artifacts", "models"),
            "checkpoint_path": os.path.join(
                self.base_dir, "artifacts", "models", "checkpoint.weights.h5"
            ),
            "best_model_path": os.path.join(
                self.base_dir, "artifacts", "models", "best_model.keras"
            ),
            "tflite_output_path": os.path.join(
                self.base_dir, "artifacts", "models", "milklife_int8.tflite"
            ),
            "log_dir": os.path.join(self.base_dir, "artifacts", "logs"),
        }
        for attr, value in defaults.items():
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, value)


# ── Global singleton instances ───────────────────────────────────────
SENSOR = SensorConfig()
TRAINING = TrainingConfig()
DATA = DataConfig()
PATHS = PathConfig()
