"""TensorFlow Lite export with full int8 post-training quantization.

Converts the trained Keras model into a fully quantized int8 TFLite
flatbuffer suitable for deployment on ARM Cortex-M microcontrollers
via TFLite-Micro. A representative dataset is provided to the
converter so it can calibrate activation ranges for each layer.
"""

import os
from typing import Callable, Generator

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import PATHS, SENSOR
from src.data_loader import generate_dataset


def _representative_dataset_gen(
    X_cal: np.ndarray,
) -> Callable[[], Generator[list[np.ndarray], None, None]]:
    """Factory that returns a representative dataset generator.

    The converter invokes this generator to collect activation
    statistics needed for int8 calibration. We use 200 samples
    (or the full calibration set if smaller) for stable quantile
    estimation.
    """
    num_cal = min(200, len(X_cal))

    def gen() -> Generator[list[np.ndarray], None, None]:
        indices = np.random.default_rng(0).choice(
            len(X_cal), size=num_cal, replace=False
        )
        for i in indices:
            sample = X_cal[i : i + 1].astype(np.float32)
            yield [sample]

    return gen


def export_tflite(model_path: str | None = None) -> str:
    """Converts a saved Keras model to a fully int8-quantized TFLite model.

    Args:
        model_path: Path to the .keras checkpoint. Defaults to the
            pipeline's best-model path.

    Returns:
        Filesystem path to the written .tflite flatbuffer.
    """
    model_path = model_path or PATHS.best_model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first."
        )

    print("\n── TFLite Export (int8 Post-Training Quantization) ──\n")

    model = keras.models.load_model(model_path)

    # Use training split as calibration data
    X_train, _, _, _ = generate_dataset()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full integer quantization — weights AND activations are int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(PATHS.tflite_output_path), exist_ok=True)
    with open(PATHS.tflite_output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  ✓ Quantized model written to: {PATHS.tflite_output_path}")
    print(f"  ✓ Model size: {size_kb:.1f} KB\n")

    _verify_tflite(PATHS.tflite_output_path)
    return PATHS.tflite_output_path


def _verify_tflite(tflite_path: str) -> None:
    """Smoke-tests the exported TFLite model with a dummy input."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("  TFLite Model Verification:")
    print(f"    Input  : {input_details['shape']}  dtype={input_details['dtype']}")
    print(f"    Output : {output_details['shape']} dtype={output_details['dtype']}")

    # Feed a zero tensor to confirm inference runs without error
    dummy = np.zeros(input_details["shape"], dtype=input_details["dtype"])
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    print(f"    Smoke-test output: {output}")
    print("  ✓ TFLite inference OK\n")


if __name__ == "__main__":
    export_tflite()
