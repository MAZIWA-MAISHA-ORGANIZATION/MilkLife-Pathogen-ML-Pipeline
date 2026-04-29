"""Training orchestration for the MilkLife Pathogen Detection model.

Handles dataset construction, callback configuration, model fitting,
and artifact persistence. Designed to be run as a standalone script
or imported as a module.
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras

from src.config import TRAINING, PATHS, DATA
from src.data_loader import generate_dataset, build_tf_dataset
from src.model import build_model, compile_model


def _ensure_directories() -> None:
    """Creates artifact directories if they do not already exist."""
    os.makedirs(PATHS.model_dir, exist_ok=True)
    os.makedirs(PATHS.log_dir, exist_ok=True)


def _build_callbacks() -> list[keras.callbacks.Callback]:
    """Constructs the callback stack for training.

    Returns:
        List of configured Keras callbacks.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=TRAINING.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=PATHS.best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAINING.reduce_lr_factor,
            patience=TRAINING.reduce_lr_patience,
            min_lr=TRAINING.min_lr,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=PATHS.log_dir,
            histogram_freq=1,
            write_graph=False,
        ),
    ]


def train() -> keras.Model:
    """Executes the full training pipeline.

    Steps:
        1. Generate synthetic dataset and split train / test.
        2. Wrap arrays in optimised tf.data pipelines.
        3. Build and compile the 1D-CNN.
        4. Fit with callbacks (EarlyStopping, Checkpoint, LR scheduler).
        5. Save the final model and return it for downstream evaluation.

    Returns:
        The trained Keras model with best-epoch weights restored.
    """
    _ensure_directories()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   MilkLife Pathogen ML Pipeline — Training      ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── Data ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = generate_dataset()

    # Carve a validation set from the training partition
    val_size = int(len(X_train) * TRAINING.validation_split)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    print(f"  Train samples : {len(X_train)}")
    print(f"  Val samples   : {len(X_val)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"  Classes       : {DATA.class_names}\n")

    train_ds = build_tf_dataset(X_train, y_train, is_training=True)
    val_ds = build_tf_dataset(X_val, y_val, is_training=False)

    # ── Model ────────────────────────────────────────────────────────
    model = build_model()
    model = compile_model(model)
    model.summary()

    # ── Fit ───────────────────────────────────────────────────────────
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING.epochs,
        callbacks=_build_callbacks(),
        verbose=1,
    )

    # Persist final model alongside the best checkpoint
    model.save(PATHS.best_model_path)
    print(f"\n✓ Model saved to {PATHS.best_model_path}")

    return model


if __name__ == "__main__":
    train()
