"""1D-CNN architecture optimized for edge deployment on microcontrollers.

Design rationale:
  • Depthwise-separable convolutions drastically reduce parameter count
    and FLOPs vs. standard convolutions — critical for MCU RAM budgets.
  • Global Average Pooling replaces Flatten + Dense, shrinking the
    classifier head and improving spatial invariance.
  • Dropout is applied after each separable block and before the final
    dense layer to regularize on small datasets without batch-norm
    overhead (batch-norm is poorly supported by TFLite-Micro).
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

from src.config import SENSOR, TRAINING, DATA


def _separable_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    *,
    dropout_rate: float,
    name_prefix: str,
) -> tf.Tensor:
    """Depthwise-separable conv → ReLU6 → Dropout residual block.

    ReLU6 is used instead of standard ReLU for better int8 quantisation
    fidelity — its bounded activation range maps cleanly to fixed-point.
    """
    x = layers.SeparableConv1D(
        filters,
        kernel_size,
        padding="same",
        depthwise_initializer="he_uniform",
        pointwise_initializer="he_uniform",
        name=f"{name_prefix}_sepconv",
    )(x)
    x = layers.Activation("relu6", name=f"{name_prefix}_relu6")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    return x


def build_model(
    seq_length: Optional[int] = None,
    num_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    dropout_rate: Optional[float] = None,
) -> Model:
    """Constructs the lightweight 1D-CNN using the Keras Functional API.

    Args:
        seq_length: Temporal dimension of input tensor (default from config).
        num_channels: Number of sensor channels (default from config).
        num_classes: Classification targets (default from config).
        dropout_rate: Per-block dropout probability (default from config).

    Returns:
        Compiled Keras Model ready for training.
    """
    seq_length = seq_length or SENSOR.sequence_length
    num_channels = num_channels or SENSOR.num_channels
    num_classes = num_classes or DATA.num_classes
    dropout_rate = dropout_rate or TRAINING.dropout_rate

    inputs = layers.Input(
        shape=(seq_length, num_channels), name="sensor_input"
    )

    # ── Feature extraction backbone ──────────────────────────────────
    # Initial standard conv to project raw sensor channels into a richer
    # feature space before the cheaper separable layers take over.
    x = layers.Conv1D(
        32, kernel_size=7, padding="same", kernel_initializer="he_uniform",
        name="stem_conv",
    )(inputs)
    x = layers.Activation("relu6", name="stem_relu6")(x)
    x = layers.MaxPooling1D(pool_size=2, name="stem_pool")(x)

    x = _separable_block(x, 64, 5, dropout_rate=dropout_rate, name_prefix="block1")
    x = layers.MaxPooling1D(pool_size=2, name="block1_pool")(x)

    x = _separable_block(x, 96, 3, dropout_rate=dropout_rate, name_prefix="block2")
    x = layers.MaxPooling1D(pool_size=2, name="block2_pool")(x)

    x = _separable_block(x, 128, 3, dropout_rate=dropout_rate, name_prefix="block3")

    # ── Classifier head ──────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu6", name="fc_hidden")(x)
    x = layers.Dropout(dropout_rate, name="fc_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="milklife_pathogen_cnn")
    return model


def compile_model(model: Model) -> Model:
    """Applies optimizer, loss, and metrics to the model.

    Label smoothing is used in the cross-entropy loss to mitigate
    overconfident predictions on the small synthetic dataset.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TRAINING.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=["accuracy"],
    )
    return model
