# Machine Learning Architecture Overview

This document defines the Machine Learning pipeline for the Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging.

It specifically documents the responsibilities and boundaries of the ML-side modules owned by Member 1, while showing how these modules connect to the later privacy and federated-learning components.

## 1. ML Architecture Overview

The hospital-side ML pipeline processes medical-imaging datasets locally before local model training and federated participation.

```text
Hospital Local Dataset
        ↓
Module 4
Dataset Ingestion and Inspection
        ↓
DatasetProfile
        ↓
Module 5
Automated Preprocessing Engine
        ↓
ML-ready Dataset / Data Pipeline
        ↓
Module 6
Model Management
        ↓
CNN Architecture / Model
        ↓
Module 7
Local Deep Learning Training
        ↓
Module 8
Resource-Aware Training
        ↓
Local Model Weights
        ↓
Differential Privacy
        ↓
Federated Learning Client
        ↓
Central Flower Aggregator
        ↓
Global Model
        ↓
Model Versioning
        ↓
Hospital Local Model
        ↓
Module 16
Local Inference
```

### Implementation Status

- [x] COMPLETED: Module 4 — Dataset Ingestion and Inspection
- [x] COMPLETED: Module 5 — Automated Preprocessing Engine
- [x] COMPLETED: Module 6 — Model Management
- [ ] NOT COMPLETED: Module 7 — Local Deep Learning Training
- [ ] NOT COMPLETED: Module 8 — Resource-Aware Training
- [ ] NOT COMPLETED: Module 16 — Local Inference

Only Modules 4 and 5 are currently marked as completed in this document.

## 2. Member 1 ML Module Scope

Member 1 is responsible for the following ML-side modules:

| Module | Name | Status |
| :--- | :--- | :--- |
| 4 | Dataset Ingestion and Inspection | [x] COMPLETED |
| 5 | Automated Preprocessing Engine | [x] COMPLETED |
| 6 | Model Management | [x] COMPLETED |
| 7 | Local Deep Learning Training | [ ] NOT COMPLETED |
| 8 | Resource-Aware Training | [ ] NOT COMPLETED |
| 16 | Local Inference | [ ] NOT COMPLETED |

These modules form the intended hospital-side ML pipeline.

The modules are intentionally separated so that dataset handling, preprocessing, model management, training, resource adaptation, and inference can evolve independently.

## 3. Module 4: Dataset Ingestion and Inspection
**Status: [x] COMPLETED**

Module 4 is the entry point for hospital-side ML data processing.

Its responsibility is to inspect and profile the locally available dataset without modifying the original source data.

### Supported Dataset Types
- **Image datasets**: JPG/JPEG, PNG, BMP, TIF/TIFF, WEBP
- **Tabular datasets**: CSV, XLS, XLSX

### Dataset Inspection Includes
- recursive file discovery
- file validation
- image readability
- image dimensions
- image channels/color modes
- dataset split detection
- class detection
- class distributions
- imbalance warnings
- tabular schema
- column types
- missing values
- duplicates
- candidate target columns
- validation warnings/errors

### Outputs
- `DatasetProfile`
- `dataset_profile.json`
- `dataset_report.md`

Module 4 does not perform:
- preprocessing
- model training
- model aggregation
- differential privacy
- federated learning
- inference

## 4. Module 4 → Module 5 Contract
**Status: [x] COMPLETED**

The primary interface between Module 4 and Module 5 is the `DatasetProfile`.

```text
Module 4
    ↓
DatasetProfile
    ↓
Module 5
```

The `DatasetProfile` provides metadata about the dataset that Module 5 can use when making preprocessing and data-quality decisions.

It may include information such as:
- dataset type
- sample counts
- detected splits
- classes
- class distributions
- image statistics
- tabular statistics
- missing-data information
- duplicates
- warnings
- errors

Module 5 may perform additional validation where required for preprocessing.

**Responsibility Boundary**
- Module 4 asks: *What is contained in this dataset?*
- Module 5 asks: *Given what was discovered, how should this dataset be safely prepared for ML?*

## 5. Module 5: Automated Preprocessing Engine
**Status: [x] COMPLETED**

Module 5 converts the validated dataset and Module 4 profile into a reproducible ML-ready data pipeline.

Its responsibilities include:
- input safety validation
- image quality validation
- tabular quality validation
- duplicate/rejection handling
- class and label hierarchy handling
- train/validation/test splitting
- patient/group-level leakage prevention where reliable identifiers exist
- class-aware splitting
- class imbalance handling
- image preprocessing
- tabular preprocessing
- training-only augmentation configuration
- reproducibility
- preprocessing reporting

### Processing Modes

Module 5 supports two conceptual output modes.

**Materialized Mode**
A separate processed dataset can be created:
```text
processed_dataset/
├── train/
├── validation/
└── test/
```
The original dataset remains untouched.

**Lazy/Manifest Mode**
For very large datasets, Module 5 can generate lightweight manifests:
- `train_manifest.json`
- `validation_manifest.json`
- `test_manifest.json`

The later training DataLoader can reference the original files and apply the required transformations when samples are loaded. This avoids unnecessarily creating another physical copy of a very large dataset.

### Large Dataset Principle
A large dataset must still be completely considered.
For example, a 100 GB dataset must not be silently reduced to a small subset merely because of its size.

Instead, Module 5 uses approaches such as:
- incremental scanning
- lazy loading
- chunked processing
- manifests/indexes
- bounded memory usage

**Important Boundary**
Module 5 does not perform model training.
It produces the data representation required by Module 7.

## 6. Module 5 → Module 7 Contract

The output of Module 5 is the input to the future local training pipeline.

**Materialized Mode**
```text
Module 5
    ↓
processed_dataset/
├── train/
├── validation/
└── test/
    ↓
Module 7
```

**Lazy Mode**
```text
Module 5
    ↓
train_manifest.json
validation_manifest.json
test_manifest.json
    ↓
Module 7 DataLoader
    ↓
Preprocessing on demand
```

Module 5 also provides relevant metadata such as:
- labels
- class mapping
- hierarchy mapping
- split information
- preprocessing configuration
- transformation parameters
- sampling strategy
- reproducibility information

## 7. Module 6: Model Management
**Status: [x] COMPLETED**

Module 6 manages the ML model architectures used by the local training pipeline.

The implemented model catalog contains eight supported architectures organized into the project's resource-aware tiers:

### High-End
- ViT-B/16
- EfficientNet-B4

### Medium-End
- EfficientNet-B0
- ResNet50

### Low-End
- ResNet18
- MobileNetV2

### Very-Low-End
- MobileNetV3-Small
- MobileViT-XXS

Module 6 is responsible for:
- model architecture registry/catalog
- model configuration
- model construction
- model metadata
- model artifact storage using safetensors
- SHA-256 artifact integrity validation
- strict model weight/key/shape validation
- local immutable model versioning
- model/preprocessing compatibility checks with Module 5
- CPU/CUDA model handling and checkpoint portability
- deterministic NumPy parameter extraction/loading for future federated integration
- controlled pretrained-weight handling without silent downloads
- model-management CLI operations

EfficientNet-B0 remains the project's primary initial CNN direction, while the complete eight-model catalog is available for later resource-aware selection.

Module 6 does not perform:
- local model training
- resource-aware model selection
- federated aggregation
- differential privacy
- secure communication
- Byzantine defense
- inference execution

Resource-aware model selection belongs to Module 8, local training belongs to Module 7, and federated aggregation belongs to Module 9.

## 8. Module 7: Local Deep Learning Training
**Status: [ ] NOT COMPLETED**

Module 7 will perform actual local deep-learning training using the hospital's local data.

Conceptually:
```text
Module 5 ML-ready data
        ↓
PyTorch Dataset/DataLoader
        ↓
Module 6 Model
        ↓
Forward Pass
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Optimizer
        ↓
Local Model Update
        ↓
Validation / Evaluation
        ↓
Local Model Weights
```

### Intended Responsibilities
Module 7 will handle:
- PyTorch DataLoader
- batching
- shuffling
- model execution
- forward pass
- loss calculation
- backpropagation
- optimizer stepping
- local epochs
- validation
- evaluation
- checkpoints where applicable
- generation of local model weights

Module 7 will not perform:
- federated aggregation
- global model aggregation
- authentication
- secure communication
- Byzantine defense

## 9. Module 8: Resource-Aware Training
**Status: [ ] NOT COMPLETED**

Module 8 will adapt local training configuration to the hardware available at the hospital.

Hospital systems may have different:
- GPUs
- GPU VRAM
- CPUs
- system RAM
- compute capabilities

Resource-aware training can consider:
- GPU availability
- CUDA availability
- available VRAM
- system RAM
- CPU availability
- batch-size constraints
- CPU fallback

The objective is to allow the local training pipeline to operate within the available hardware resources.

Conceptually:
```text
Hospital Hardware
        ↓
Module 8
Resource Detection
        ↓
Training Configuration
        ↓
Module 7
Local Training
```
Module 8 will not perform federated aggregation.

## 10. Member 1 ML Training Flow

The intended complete Member 1 ML pipeline is:

```text
                    HOSPITAL
                       │
                       ▼
              Local Dataset
                       │
                       ▼
          ┌─────────────────────┐
          │ Module 4            │
          │ Dataset Inspection   │
          └─────────────────────┘
                       │
                       ▼
                DatasetProfile
                       │
                       ▼
          ┌─────────────────────┐
          │ Module 5            │
          │ Preprocessing       │
          └─────────────────────┘
                       │
                       ▼
             ML-ready Dataset
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      ┌─────────────┐     ┌─────────────┐
      │ Module 6    │     │ Module 8    │
      │ Model       │     │ Resources   │
      │ Management  │     │             │
      └──────┬──────┘     └──────┬──────┘
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌────────────────┐
              │ Module 7       │
              │ Local Training │
              └───────┬────────┘
                      │
                      ▼
              Local Model Weights
                      │
                      ▼
             Later Privacy/FL Modules
```

## 11. Privacy Boundary

The ML pipeline operates on hospital-local data.

The fundamental privacy principle is:
**Raw medical datasets remain within the hospital-side environment.**

Therefore:
- raw medical images remain local
- raw patient records remain local
- Module 4 operates locally
- Module 5 operates locally
- future Module 7 will perform local training
- future Module 8 will operate locally
- future Module 16 will perform local inference

Later federated-learning components communicate protected model information rather than raw medical images.

Federated learning does not mean that absolutely nothing leaves the hospital. Model updates and required system information may leave the hospital and therefore require appropriate privacy and security mechanisms in later modules.

## 12. Module 16: Local Inference
**Status: [ ] NOT COMPLETED**

Module 16 will provide local model inference at the hospital.

The approved model will be loaded within the hospital-side environment.

Conceptually:
```text
Approved Model
      ↓
Module 16
      ↓
New Local Medical Image
      ↓
Module 5 Preprocessing
      ↓
Model Input
      ↓
CNN
      ↓
Prediction
```

### Privacy Behavior
New patient images remain local. The image will not be sent to the central federated aggregation service merely to obtain a prediction.

Module 5 preprocessing routines can be reused locally so that inference input processing remains consistent with the model's expected preprocessing configuration.

### Important Limitation
Module 16 will produce model predictions. It must not be described as guaranteeing clinical diagnosis or clinical validity.

## 13. Member 1 Module Boundaries

| Module | Responsibility | Location | Main Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| Module 4 | Dataset ingestion, validation and profiling | Hospital | DatasetProfile | [x] COMPLETED |
| Module 5 | Automated preprocessing and ML-ready data preparation | Hospital | Processed dataset / manifests / preprocessing metadata | [x] COMPLETED |
| Module 6 | Model architecture and model management | Hospital | PyTorch model/configuration | [ ] NOT COMPLETED |
| Module 7 | Local deep-learning training and evaluation | Hospital | Local model weights/checkpoints | [ ] NOT COMPLETED |
| Module 8 | Hardware/resource-aware training configuration | Hospital | Resource-aware training parameters | [ ] NOT COMPLETED |
| Module 16 | Local model inference | Hospital | Model predictions | [ ] NOT COMPLETED |

## 14. Relationship with Other Project Modules

Member 1's modules connect with other project modules but do not absorb their responsibilities.

- **Module 1**: Authentication and role management. Outside Member 1 ML implementation scope.
- **Module 9**: Federated Learning Engine. Receives local model updates and handles federated orchestration.
- **Module 10**: Differential Privacy. Protects local model information before federation.
- **Module 11**: Secure Communication. Protects communication between hospital and central services.
- **Module 12**: Byzantine/Malicious Update Detection. Handles suspicious or malicious federated model updates.
- **Module 13**: Monitoring and Telemetry. Handles system/training monitoring.
- **Module 14**: Database, Audit and Model Storage. Handles persistent system metadata and storage infrastructure.
- **Module 15**: Model Versioning. Tracks model versions/checkpoints across the broader system.

## 15. ML Component Boundary

```text
┌───────────────────────────────────────────────┐
│              HOSPITAL ENVIRONMENT             │
│                                               │
│  Module 4                                     │
│  Dataset Ingestion & Inspection               │
│               ↓                               │
│       DatasetProfile                          │
│               ↓                               │
│  Module 5                                     │
│  Automated Preprocessing                      │
│               ↓                               │
│       ML-ready Dataset                        │
│               ↓                               │
│  Module 6                                     │
│  Model Management                             │
│               ↓                               │
│  Module 8                                     │
│  Resource-Aware Configuration                 │
│               ↓                               │
│  Module 7                                     │
│  Local Deep Learning Training                 │
│               ↓                               │
│       Local Model Weights                     │
│               ↓                               │
│       Later Privacy / FL Modules              │
│                                               │
│  Module 16                                    │
│  Local Inference                              │
│               ↑                               │
│       Approved Model                          │
│                                               │
└───────────────────────────────────────────────┘
                       │
                       │ Protected model information
                       ▼
              Federated Components
```

## 16. Design Principles

- **Raw-Data Locality**: Raw medical datasets remain within the hospital-side environment and are not transmitted to the central federated aggregation service.
- **Non-Destructive Processing**: The original dataset remains untouched. Module 5 creates derived outputs only when configured to do so.
- **Complete Dataset Consideration**: Large datasets must not be silently reduced to arbitrary subsets because of their size. Resource-efficient processing should be used instead.
- **Reproducibility**: Dataset profiles, preprocessing configurations, split decisions, sampling strategies, transformation parameters, model configurations, and training checkpoints should support reproducible ML experiments.
- **Modularity**: Each Member 1 module has a defined responsibility:
  - M4 → What data exists?
  - M5 → How should it be prepared?
  - M6 → What model should process it?
  - M7 → How should it be trained?
  - M8 → How should training adapt to hardware?
  - M16 → How should the trained model be used locally?
- **Privacy Preservation**: Raw medical data remains local. Privacy protection of model information is handled by the appropriate later privacy/federated modules.
- **Extensibility**: The architecture should allow additional compatible CNN architectures and preprocessing configurations without redesigning the entire ML pipeline.

## 17. Current Implementation Status

The current Member 1 ML module status is:

- [x] Module 4 — Dataset Ingestion and Inspection
- [x] Module 5 — Automated Preprocessing Engine
- [x] Module 6 — Model Management
- [ ] Module 7 — Local Deep Learning Training
- [ ] Module 8 — Resource-Aware Training
- [ ] Module 16 — Local Inference

Modules 4, 5, and 6 are currently marked as completed.

### Remaining Task

- [ ] Module 4 future enhancement: add standardized existing-split validation/preservation for tabular datasets (CSV/XLS/XLSX) when a hospital provides an already-split tabular dataset. This is a future enhancement and does not make the currently completed Module 4 implementation incomplete.

