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
Module 8
Resource-Aware Training
(future configuration provider)
        ↓
Module 7
Local Deep Learning Training
        ↓
Local Model Weights
+ FederationHandoff
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
- [x] COMPLETED: Module 7 — Local Deep Learning Training
- [ ] NOT COMPLETED: Module 8 — Resource-Aware Training
- [ ] NOT COMPLETED: Module 16 — Local Inference

Modules 4, 5, 6, and 7 are currently marked as completed in this document.

## 2. Member 1 ML Module Scope

Member 1 is responsible for the following ML-side modules:

| Module | Name | Status |
| :--- | :--- | :--- |
| 4 | Dataset Ingestion and Inspection | [x] COMPLETED |
| 5 | Automated Preprocessing Engine | [x] COMPLETED |
| 6 | Model Management | [x] COMPLETED |
| 7 | Local Deep Learning Training | [x] COMPLETED |
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
**Status: [x] COMPLETED**

Module 7 is the implemented hospital/local-side PyTorch training engine. It consumes
Module 5's prepared image-classification manifests, obtains a model from Module 6,
trains it locally, evaluates it, tracks metrics, checkpoints the best result, and
prepares a validated `FederationHandoff` for the future Module 9 federated-learning
engine. Module 7 does not transmit or aggregate model updates.

Conceptually:
```text
Module 5 ML-ready data / manifests
        ↓
PyTorch Dataset / DataLoader
        ↓
Module 6 Model
        ↓
Optional Module 8 resolved configuration
        ↓
Forward Pass
        ↓
Loss Calculation
        ↓
Backpropagation + Optimizer
        ↓
Optional LR Scheduler
        ↓
Validation / Evaluation
        ↓
Best Model Checkpoint
        ↓
TrainingResult
        ↓
Validated FederationHandoff
```

### Implemented Responsibilities

Module 7 handles:
- PyTorch Dataset/DataLoader construction from Module 5 manifests
- batching and deterministic shuffling
- reproducible DataLoader worker RNG seeding
- all eight Module 6 architectures through one common `Trainer`
- forward pass, loss calculation, backpropagation, and optimizer stepping
- local training epochs and validation/evaluation
- accuracy, precision, recall, F1, per-class metrics, and confusion matrices
- configurable early stopping and best-checkpoint selection
- checkpoint/resume of optimizer, scheduler, GradScaler, epoch, bookkeeping, and RNG state
- CUDA OOM handling without silent device/model fallback
- optional mixed precision: `fp32`, `fp16`, and `auto`
- LR schedulers: `none`, `reduce_on_plateau`, `step`, and `cosine`
- structured `TrainingResult`
- versioned `FederationHandoff` generation with parameter metadata and SHA-256 integrity validation
- CLI support for precision, scheduler, worker-count, training, and validation configuration

### Mixed Precision

`precision` supports `fp32` (default), `fp16`, and `auto`. Precision is resolved once
against the already-selected device. `fp16` requires CUDA; `auto` selects FP16 on CUDA
and FP32 on CPU without changing the device. PyTorch's current AMP API and
`GradScaler` are used, and scaler state is persisted across resume.

### Learning-Rate Scheduling

The common trainer supports:
- `none` - default, unchanged behavior
- `reduce_on_plateau` - stepped on validation loss
- `step` - `StepLR`, stepped once per epoch
- `cosine` - `CosineAnnealingLR`, stepped once per epoch

Scheduler state is included in the private training-state checkpoint and restored on
resume.

### DataLoader Worker Reproducibility

A `torch.Generator` controls shuffling, while worker processes receive deterministic
Python, NumPy, and PyTorch seeds derived from the configured random seed and worker ID.
The worker initializer is module-level so it remains picklable under Windows `spawn`.

This establishes reproducible shuffling and worker-local RNG state for the same seed,
worker count, and software/hardware stack. It does not claim bit-for-bit
cross-platform/GPU determinism.

### FederationHandoff Boundary

`FederationHandoff` is a versioned Module 7 → Module 9 contract. It contains model
identity, ordered parameters, parameter metadata, training/validation metrics,
configuration, resolved device/precision, and optional caller-supplied `round_id` and
`client_id`. Parameters are stored in `.npz`; metadata is stored separately in JSON.
A SHA-256 checksum detects parameter corruption or tampering.

Module 7 does not implement Flower, FedAvg/FedProx, network transport, aggregation,
authentication, differential privacy, or secure communication. Those responsibilities
remain with the later modules.

### Verification

The implemented Module 7 test suite contains **165 passing tests**, and the full
`hospital_client/` suite contains **414 passing tests with 0 failures**. Training
coverage is **98% (1774 statements, 39 missed)**.

Additional real-hardware verification included:
- CPU + FP32 + cosine scheduler + `num_workers=2`
- CUDA on an NVIDIA GeForce RTX 3050 Laptop GPU + FP16 + StepLR
- real AMP/autocast and GradScaler execution
- valid FederationHandoff generation and validation
- save/load round-trip with byte-identical parameter arrays

### Scope and Limitations

Module 7 is **single-GPU only** as currently implemented. No throughput/memory
benchmark was performed for FP16, so no performance claim is made. Tests use tiny
synthetic datasets for functional verification and therefore do not establish
medical-imaging accuracy or clinical validity. Formal research experiments belong to
Module 20.

Module 7 does not yet receive a federated global model through a Module 9 round-trip
protocol, and it does not send completion/failure notifications. Those are future
integrations.

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
- Module 7 performs local training
- future Module 8 will operate locally
- future Module 16 will perform local inference

Module 7 prepares a validated `FederationHandoff`, but does not transmit it. Later federated-learning, privacy, and security components handle protected model information rather than raw medical images.

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
| Module 6 | Model architecture and model management | Hospital | PyTorch model/configuration | [x] COMPLETED |
| Module 7 | Local deep-learning training and evaluation | Hospital | TrainingResult / checkpoints / FederationHandoff | [x] COMPLETED |
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
│  Resource-Aware Configuration (future)        │
│               ↓                               │
│  Module 7                                     │
│  Local Deep Learning Training                 │
│               ↓                               │
│       Local Model Weights                     │
│               ↓                               │
│       FederationHandoff                        │
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
- [x] Module 7 — Local Deep Learning Training
- [ ] Module 8 — Resource-Aware Training
- [ ] Module 16 — Local Inference

Modules 4, 5, 6, and 7 are currently marked as completed.

### Remaining Tasks / Future Work

The following items are the remaining enhancements and integration work
identified for the completed Module 4-7 implementation. These items do not
mean that Modules 4, 5, 6, or 7 are incomplete.

#### 1. Module 4 — Existing-Split Validation and Preservation

- [ ] **Add standardized existing-split validation/preservation for tabular
  datasets (CSV/XLS/XLSX).**

When a hospital provides a tabular dataset that is already divided into
training, validation, and test sets, Module 4 should recognize the existing
split structure instead of unnecessarily creating a new split. The
enhancement should validate that the supplied partitions are present,
structurally valid, and usable by downstream modules, while preserving the
hospital-provided split decisions. The validated split information should be
passed consistently to Module 5 and the later training pipeline.

This is a future enhancement to the completed Module 4 implementation. It
does not make the current Module 4 implementation incomplete.

#### 2. Module 18 — Notification Integration

- [ ] **Notification integration (Module 18 - see §21)**

Module 7 already produces a structured `TrainingResult` representing training
completion, failure, or interruption. It also produces a validated
`FederationHandoff` when the training run generates a handoff artifact.
The remaining work is to connect these outputs to the platform's notification
layer.

Module 18 should consume the training status and relevant result information
and deliver appropriate notifications to the intended researcher, hospital
operator, or user interface. Notifications may cover events such as training
completion, training failure, and training interruption. The notification
delivery mechanism itself is outside Module 7.

#### 3. Module 20 — Medical Accuracy / Research Experiments

- [ ] **Medical accuracy / research experiments (Module 20 - see §21)**

Module 7 calculates standard classification metrics including accuracy,
precision, recall, F1, per-class metrics, and a confusion matrix for the
dataset supplied to it. However, its functional tests use small synthetic
datasets and are intended to establish software correctness rather than real
medical-imaging accuracy.

The remaining research work is therefore to perform formal experiments on
appropriate real medical-imaging datasets, compare the supported models and
training configurations, establish suitable experimental or clinical
baselines, perform benchmarking, analyze the resulting metrics, and document
the findings in a research-oriented manner. Any discussion of medical
accuracy, clinical relevance, or model superiority should be based on these
formal experiments rather than Module 7's functional test results.

#### 4. Mixed-Precision Performance Benchmarking

- [ ] **Benchmark FP32 versus FP16 mixed-precision training on CUDA hardware.**

Module 7 has verified that FP16 mixed-precision training works end-to-end on
real CUDA hardware, but a throughput and memory benchmark was not performed.
The remaining work is to measure training time, throughput, and GPU memory
usage for comparable FP32 and FP16 configurations and document the results.
The benchmark should be treated as an experimental measurement rather than
assuming a speed-up or memory reduction.

#### 5. Cross-Environment Reproducibility Validation

- [ ] **Extend reproducibility validation beyond the same
  machine/software-stack configuration.**

Module 7's deterministic DataLoader worker seeding provides reproducible
worker behavior when the seed and worker count are kept consistent on the
same machine and software stack. Further validation can examine behavior
across different operating systems, hardware configurations, PyTorch/CUDA
versions, and other environments. The purpose is to document the practical
reproducibility boundary rather than claim universal bit-for-bit
determinism.

#### 6. Extended Training and Checkpoint Validation

- [ ] **Run additional end-to-end training experiments across the supported
  configurations.**

The implementation should be exercised with longer training runs and a
broader combination of model architectures, precision modes, learning-rate
scheduler configurations, worker counts, and checkpoint/resume scenarios.
This provides additional validation beyond the small synthetic functional
tests and helps identify configuration-specific issues before broader
platform integration.

#### 7. Federation Handoff Integration Validation

- [ ] **Validate consumption of the Module 7 `FederationHandoff` by the
  subsequent federated workflow.**

Module 7 already creates and validates the versioned handoff contract, but
the completed Module 7 implementation does not itself implement the
federated round-trip in which a global model is received and subsequently
used for local training. The remaining integration work is to verify that
the handoff artifact can be consumed correctly by the later federated
workflow, including parameter files, metadata, protocol version, client and
round identifiers, and checksum validation.

The federated orchestration and global-model round-trip remain outside the
responsibility of Module 7.
