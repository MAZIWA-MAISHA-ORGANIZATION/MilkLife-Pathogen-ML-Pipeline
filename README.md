# MilkLife Pathogen ML Pipeline

> **Core Developer:** Steve Austine Kamunge
> **Role:** Lead AI Engineer (Architected the complete Python Machine Learning pipeline, handled multi-channel data processing, and built the edge-deployment infrastructure).

---

## Project Overview

**MilkLife** is an IoT diagnostic probe that detects dairy pathogens (*E. coli*, *S. aureus*) and early-stage spoilage in raw milk using multi-channel sensor data. The probe fuses three analog channels — **ISFET pH**, **conductivity (µS/cm)**, and **RGB colorimetric intensity** — into a single inference pass on a microcontroller.

This repository contains the **Python Machine Learning pipeline** used to:

1. **Generate & augment** synthetic multi-channel sensor datasets.
2. **Train** a lightweight 1D Convolutional Neural Network optimised for ARM Cortex-M targets.
3. **Evaluate** model quality with production-grade metrics (Confusion Matrix, F1-Score, ROC-AUC).
4. **Export** a fully int8-quantised TensorFlow Lite model ready for deployment via TFLite-Micro.

> **Note:** All C++ firmware and on-device inference code is maintained in a separate hardware repository.

---

## Architecture

```
MilkLife-Pathogen-ML-Pipeline/
├── src/
│   ├── __init__.py            # Package initialiser
│   ├── config.py              # Centralised hyperparameters & path constants
│   ├── data_loader.py         # Synthetic data generation & tf.data pipeline
│   ├── model.py               # 1D-CNN (separable convolutions, Functional API)
│   ├── train.py               # Training loop with custom callbacks
│   ├── evaluate.py            # Confusion matrix, F1, ROC-AUC evaluation
│   └── export_tflite.py       # Int8 post-training quantisation → .tflite
├── artifacts/                 # Auto-generated models & logs (git-ignored)
├── requirements.txt
└── README.md
```

### Module Breakdown

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclass singletons for sensor dimensions, training hyperparameters, data config, and filesystem paths. Single source of truth. |
| `data_loader.py` | Generates class-conditional synthetic signals with domain-aware profiles (pH drift, conductivity variance). Wraps data in `tf.data.Dataset` with Gaussian noise and time-shift augmentation. |
| `model.py` | Builds a lightweight 1D-CNN using the Keras Functional API. Uses depthwise-separable convolutions and ReLU6 activations for optimal int8 quantisation fidelity. Global Average Pooling minimises the classifier head. |
| `train.py` | Orchestrates dataset creation, model compilation, and fitting. Configures `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and `TensorBoard` callbacks. |
| `evaluate.py` | Loads the best checkpoint and computes a full confusion matrix, per-class and macro F1-Score, and one-vs-rest macro ROC-AUC on the held-out test set. |
| `export_tflite.py` | Converts the trained `.keras` model into a **fully int8-quantised** `.tflite` flatbuffer using a representative calibration dataset. Includes a smoke-test to verify inference. |

---

## Model Design Rationale

| Design Choice | Why |
|---|---|
| **Separable 1D Convolutions** | ~8× fewer FLOPs than standard convolutions — critical for MCU RAM/flash budgets. |
| **ReLU6 Activation** | Bounded output `[0, 6]` maps cleanly to int8 fixed-point, reducing quantisation error vs. unbounded ReLU. |
| **Global Average Pooling** | Replaces `Flatten → Dense`, drastically cutting parameter count and improving spatial invariance. |
| **No Batch Normalisation** | Poorly supported by TFLite-Micro at inference time; dropout provides sufficient regularisation. |
| **Int8 Post-Training Quantisation** | Compresses the model by ~4× and enables integer-only inference on Cortex-M devices without an FPU. |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/MAZIWA-MAISHA-ORGANIZATION/MilkLife-Pathogen-ML-Pipeline.git
cd MilkLife-Pathogen-ML-Pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Training Pipeline

```bash
# Train the model
python -m src.train

# Evaluate on the test set
python -m src.evaluate

# Export to quantised TFLite
python -m src.export_tflite
```

All artifacts (trained model, TFLite flatbuffer, TensorBoard logs) are written to `artifacts/`.

### Monitor Training

```bash
tensorboard --logdir artifacts/logs
```

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
