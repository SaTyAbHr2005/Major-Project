# Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging

## Overview

This repository contains the Hospital Client portion of a secure and privacy-preserving federated deep learning project for medical imaging.

The client application focuses on the stages that occur before machine-learning execution, with particular emphasis on:
- secure handling of localized medical datasets
- patient privacy protection
- read-only source-data handling
- dataset validation
- dataset profiling
- automated preprocessing
- data-quality assessment
- leakage prevention
- class balancing
- reproducible preprocessing
- controlled hand-off to the downstream training engine

The repository is organized into isolated modules so that each stage has a clear responsibility and boundary. This separation is intended to improve privacy, stability, maintainability, testing, and reproducibility.

## 1. Project Goals

The Hospital Client is designed to provide a controlled local data-processing environment before data is used by the machine-learning and federated-learning stages.

The core goals are:
- Protect source data by treating localized medical datasets as read-only inputs.
- Protect patient privacy through local privacy and PHI checks.
- Validate datasets before they enter downstream processing.
- Profile datasets so later modules have structured information about the available data.
- Preprocess data safely without modifying the original host dataset.
- Prevent data leakage when fitting preprocessing transformations.
- Prevent patient-level leakage when patient/group identifiers are available.
- Track rejected samples without silently deleting them.
- Support scalable preprocessing through lazy and materialized execution modes.
- Provide reproducible outputs and clear module-to-module contracts.
- Maintain module isolation so later development does not unnecessarily modify completed upstream modules.

## 2. High-Level Architecture

The complete federated-learning scope is divided into modules.

For the complete Modules 1 through 8 architecture and federated-learning loop, refer to:
[`ml_pipeline_architecture.md`](./ml_pipeline_architecture.md)

The current repository documentation explicitly covers the completed Hospital Client preprocessing stages:

```text
Localized Medical Dataset
          │
          ▼
┌─────────────────────────────┐
│ Module 4                    │
│ Dataset Ingestion &         │
│ Validation Engine           │
└──────────────┬──────────────┘
               │
               │ Dataset + DatasetProfile
               ▼
┌─────────────────────────────┐
│ Module 5                    │
│ Automated Preprocessing     │
│ Engine                      │
└──────────────┬──────────────┘
               │
               │ ML-ready artifacts
               ▼
        Downstream Training
        / Federated Learning
```

Module 4 establishes the validated and profiled dataset boundary.
Module 5 consumes that information and produces the prepared artifacts required by the downstream training stage.

## 3. Project Structure

A simplified view of the Hospital Client is:

```text
Major-Project/
│
├── hospital_client/
│   │
│   ├── dataset/
│   │   ├── models/
│   │   │   └── dataset_profile.py
│   │   ├── ingestion/
│   │   │   ├── dataset_detector.py
│   │   │   └── label_detection.py
│   │   └── tests/
│   │
│   └── preprocessing/
│       ├── config.py
│       ├── engine.py
│       ├── cli.py
│       ├── __main__.py
│       ├── labels/
│       ├── image/
│       ├── tabular/
│       ├── quarantine/
│       ├── splitting/
│       ├── balancing/
│       ├── output/
│       ├── reporting/
│       └── tests/
│
├── module4_readme.md
├── module5_readme.md
├── ml_pipeline_architecture.md
├── requirements.txt
└── README.md
```

The exact repository contents may expand as subsequent modules are implemented.

## 4. Module Isolation

The project uses isolated modules with explicit responsibilities.

The current completed modules documented in this repository are:

```text
Module 4
    Dataset Ingestion and Validation
              │
              ▼
Module 5
    Automated Preprocessing
              │
              ▼
Downstream ML Training
```

This separation allows each module to be implemented and tested independently while maintaining a defined interface between stages.

## 5. Module 4: Dataset Ingestion and Validation Engine

**Status:** Completed

**Responsibility**
Module 4 safely scans and validates localized medical datasets, including image and tabular data. It establishes the initial trusted dataset boundary used by subsequent processing.

**Core Features**
- **Read-Only File Ingestion:** The source dataset is handled using strict read-only principles. Module 4 does not intentionally modify the original source files during ingestion and validation.
- **Patient Privacy:** The module includes patient privacy and PHI checking as part of its validation responsibilities. This is important because the project works with medical datasets where privacy must be considered before downstream machine-learning processing.
- **Format Compliance:** Input files are checked for supported formats and basic compliance.
- **Duplicate Detection:** Duplicate detection is supported using hashing-based mechanisms.
- **Dataset Profile Generation:** Module 4 generates a structured DatasetProfile describing the detected dataset. This profile becomes an important input to Module 5.

**Documentation**
For the detailed Module 4 design, CLI usage, validation behavior, and technical boundaries, see:
[`module4_readme.md`](./module4_readme.md)

## 6. Module 4 → Module 5 Contract

Module 5 does not independently replace the dataset discovery and profiling responsibilities of Module 4.

Instead, the flow is:

```text
Module 4
   │
   ├── Dataset detection
   ├── Dataset validation
   ├── Privacy checks
   ├── Duplicate detection
   └── DatasetProfile
            │
            ▼
Module 5
   │
   ├── Quality validation
   ├── Quarantine
   ├── Splitting
   ├── Leakage-safe transformations
   ├── Balancing
   └── Output generation
```

This provides a clear boundary between dataset ingestion/profile generation and automated preprocessing.

## 7. Module 5: Automated Preprocessing Engine

**Status:** Completed

**Responsibility**
Module 5 transforms the validated/profiled dataset into an ML-ready preprocessing state without modifying the original host data. It consumes the dataset and profile produced by Module 4.

**Core Features**
- **Scalable Output Modes:** Module 5 provides two primary output modes:
  - `lazy`
  - `materialized`

**Lazy Mode**
Lazy mode generates lightweight manifests:
- `train_manifest.json`
- `validation_manifest.json`
- `test_manifest.json`

The manifests provide downstream processing with references to the required source files and associated preprocessing metadata. This approach avoids unnecessary physical duplication of large datasets.

**Materialized Mode**
Materialized mode physically creates processed output in a separate output location. It also performs explicit storage-safety checks before large materialization operations.

## 8. Module 5 Data Quality and Quarantine

Module 5 performs localized quality checks before preprocessing.

**Image Validation**
The image pipeline supports checks including:
- corrupt or unreadable image detection
- zero-byte file detection
- extreme brightness detection
- Laplacian variance-based blur detection

Invalid samples can be rejected without modifying the original files.

**Tabular Validation**
The tabular pipeline supports:
- severe missingness detection
- configurable validation rules
- IQR-based outlier detection

**Quarantine**
Rejected samples are tracked through a quarantine registry.

The design is non-destructive:
```text
Rejected Sample
      │
      ├── Original file remains unchanged
      │
      └── Rejection reason is recorded
```
This prevents rejected samples from silently disappearing from the processing history.

## 9. Module 5 Leakage Prevention

A major requirement of the preprocessing engine is preventing information from validation and test datasets from influencing fitted preprocessing parameters.

For tabular preprocessing, fitting occurs using the training split only.

Conceptually:
```text
Training Data
     │
     ▼
FIT preprocessing parameters
     │
     ├── Imputation
     ├── Encoding
     └── Scaling
     │
     ▼
Frozen parameters
     │
     ├───────────────┐
     ▼               ▼
Validation          Test
TRANSFORM           TRANSFORM
```

Supported preprocessing behavior includes:
- mean imputation
- median imputation
- mode imputation
- constant imputation where configured
- categorical encoding
- standard scaling
- min-max scaling

Validation and test data are transformed using the training-fitted parameters rather than contributing to the fitting process.

## 10. Patient-Level Leakage Prevention

Where a patient or group identifier is available, Module 5 supports grouped splitting.

The pipeline supports:
- `StratifiedShuffleSplit`
- `GroupShuffleSplit`

The grouped strategy is important for datasets where multiple samples belong to the same patient.

For example:
```text
Patient A
├── Image 1
├── Image 2
└── Image 3
```

The grouping mechanism prevents samples belonging to the same group from being distributed across incompatible dataset splits. This helps prevent patient-level information leakage between training and evaluation data.

## 11. Dataset Splitting

Module 5 supports reproducible train/validation/test splitting.
The splitting process can use a configured random seed.
Where class distributions need to be preserved, stratified splitting is used.
If the data makes stratification impossible, fallback handling is available for applicable cases.

The resulting logical datasets are:
- Train
- Validation
- Test

The validation and test datasets remain evaluation datasets and are not artificially balanced for training purposes.

## 12. Class Balancing

Module 5 supports training-oriented class balancing.

The implemented balancing functionality includes strategies such as:
- oversampling
- undersampling
- class-weight strategies

The key boundary is:
Balancing is a training-data concern. Validation and test distributions should not be modified simply to improve training class balance.

## 13. Image Transformations and Augmentation

Module 5 separates deterministic preprocessing from random training augmentation.

Deterministic image transformations can include operations such as:
- resizing
- image format normalization
- RGBA-to-RGB conversion

Random training augmentation is not permanently baked into the source dataset. Instead, augmentation information can be retained as metadata for the downstream training stage.

```text
Module 5
   │
   ├── Deterministic transforms
   │       └── May be materialized
   │
   └── Random augmentation metadata
              │
              ▼
          Module 7 / Training
```
This separation allows runtime augmentation during model training.

## 14. Lazy Preprocessing

Lazy mode is intended for scenarios where physical duplication of a large dataset is undesirable.

Example:
```bash
python -m hospital_client.preprocessing preprocess \
    /path/to/dataset \
    --profile /path/to/dataset_profile.json \
    --output /path/to/output_dir \
    --mode lazy
```

Typical generated artifacts include:
```text
output/
├── train_manifest.json
├── validation_manifest.json
├── test_manifest.json
└── preprocessing metadata/reports
```

The source dataset remains at its original location.

## 15. Materialized Preprocessing

Materialized mode creates a physical processed representation.

A typical logical layout is:
```text
output/
├── train/
├── validation/
├── test/
├── train.csv
├── validation.csv
├── test.csv
└── metadata / manifests
```
The exact output depends on the dataset type and configured preprocessing operations.

## 16. Storage Safety

Materialized processing can require significant additional disk space.
Module 5 therefore performs storage checks before large materialization operations.
The materializer uses `psutil` to inspect disk availability and applies a configurable safety buffer.

Conceptually:
```text
Estimated output size
        +
Safety buffer
        ↓
Required available space
        ↓
Compare with available disk
        │
   ┌────┴────┐
   ▼         ▼
Enough     Not enough
  │            │
  ▼            ▼
Proceed      Abort safely
```
This reduces the risk of exhausting the host machine's storage during preprocessing.

## 17. Reporting and Reconciliation

Module 5 generates preprocessing reports containing information about the processing run.

Relevant information can include:
- discovered sample counts
- accepted sample counts
- rejected sample counts
- split counts
- preprocessing decisions
- output information
- transformation metadata

The core reconciliation principle is:
`discovered_samples = accepted_samples + rejected_samples + review_samples` (when a review state is part of the configured workflow).

The purpose is to prevent silent sample loss.

## 18. CLI Usage

The preprocessing engine provides a command-line interface through:
- `hospital_client/preprocessing/__main__.py`
- `hospital_client/preprocessing/cli.py`

General syntax:
```bash
python -m hospital_client.preprocessing preprocess \
    <dataset_path> \
    --profile <profile_path> \
    --output <output_dir> \
    --mode <lazy|materialized>
```

Example:
```bash
python -m hospital_client.preprocessing preprocess \
    ./data \
    --profile ./data/dataset_profile.json \
    --output ./processed \
    --mode lazy
```

The CLI is intended to make preprocessing repeatable and easy to integrate with later pipeline stages.

## 19. Module 5 → Training Contract

Module 5 is the preparation boundary immediately before ML training.
It explicitly does not perform:
- model training
- Federated Learning (FL)
- Differential Privacy (DP)

**Materialized Mode**
The downstream training stage receives:
- Processed train dataset
- Processed validation dataset
- Processed test dataset
- Transformation metadata
- Output/manifests metadata

**Lazy Mode**
The downstream training stage receives:
- `train_manifest.json`
- `validation_manifest.json`
- `test_manifest.json`
- Transformation metadata
- Augmentation metadata
- Source file paths

This allows the training pipeline to load data according to the selected processing mode.

## 20. Data Safety Principles

The Hospital Client follows several important data-safety principles.

- **Source Data Is Non-Destructive:** The original dataset is treated as read-only. Processing should operate on separate output locations where transformed data needs to be physically generated.
- **Rejected Data Is Not Silently Deleted:** Invalid samples are recorded through quarantine rather than simply removed from the source dataset.
- **Transformations Are Leakage-Safe:** Training data is used to fit preprocessing parameters. Validation and test data are transformed using those already-fitted parameters.
- **Patient Groups Are Protected:** When group identifiers are available, grouped splitting can prevent samples from the same patient/group from crossing dataset boundaries.

## 21. Testing and Verification

Both Module 4 and Module 5 contain automated tests.

The test strategy covers:
- normal processing paths
- edge cases
- invalid data
- splitting behavior
- preprocessing behavior
- storage safety
- output generation
- regression behavior

For the currently verified Module 5 implementation:
- Total tests: 62
- Passed: 62
- Failed: 0
- Pass rate: 100%
- Preprocessing coverage: 94%

The verified suite contains:
- Module 4 regression tests: 21
- Module 5 tests: 41
- Total: 62

The Module 4 regression suite remained stable during Module 5 verification.

The current Module 5 test coverage is 94%. Coverage measures the amount of executable code exercised by automated tests; it is not a correctness percentage.

## 22. Development and Validation Workflow

A typical development workflow for the Hospital Client is:

```text
1. Prepare localized dataset
             │
             ▼
2. Module 4 ingestion and validation
             │
             ▼
3. Generate DatasetProfile
             │
             ▼
4. Module 5 preprocessing
             │
             ├── Quality checks
             ├── Quarantine
             ├── Dataset splitting
             ├── Leakage-safe fitting
             ├── Balancing
             └── Output generation
             │
             ▼
5. Validate manifests / processed outputs
             │
             ▼
6. Run automated tests
             │
             ▼
7. Hand off artifacts to downstream training
```

## 23. Getting Started

### 23.1 Create a Virtual Environment
Create a clean Python virtual environment:
```bash
python -m venv venv
```

Activate it on Windows:
```powershell
venv\Scripts\Activate.ps1
```
Or on a Unix-like shell:
```bash
source venv/bin/activate
```

### 23.2 Install Dependencies
Install the repository dependencies:
```bash
pip install -r requirements.txt
```

### 23.3 Run the Test Suite
Run the complete Hospital Client test suite:
```bash
python -m pytest hospital_client/ -v
```

For Module 5 and its Module 4 regression tests specifically:
```bash
python -m pytest hospital_client/preprocessing/tests/ hospital_client/dataset/tests/ -v
```

To measure preprocessing coverage:
```bash
python -m pytest \
    hospital_client/preprocessing/tests/ \
    hospital_client/dataset/tests/ \
    -v \
    --cov=hospital_client.preprocessing
```

## 24. Running Module 5

A typical lazy preprocessing execution is:
```bash
python -m hospital_client.preprocessing preprocess \
    /path/to/dataset \
    --profile /path/to/dataset_profile.json \
    --output /path/to/output_dir \
    --mode lazy
```

For materialized processing:
```bash
python -m hospital_client.preprocessing preprocess \
    /path/to/dataset \
    --profile /path/to/dataset_profile.json \
    --output /path/to/output_dir \
    --mode materialized
```

The dataset profile supplied to Module 5 should be the profile generated by the upstream dataset-processing stage.

## 25. Documentation

The repository currently provides module-specific documentation:

**Module 4**
[`module4_readme.md`](./module4_readme.md)
Contains the detailed documentation for dataset ingestion, validation, privacy checks, duplicate detection, profile generation, CLI usage, and technical boundaries.

**Module 5**
[`module5_readme.md`](./module5_readme.md)
Contains the detailed documentation for automated preprocessing, quality validation, quarantine, splitting, leakage prevention, balancing, output modes, storage safety, reporting, testing, and the Module 7 training contract.

**Overall Architecture**
[`ml_pipeline_architecture.md`](./ml_pipeline_architecture.md)
Contains the high-level architecture and complete Modules 1 through 8 federated-learning scope.

## 26. Module Status

| Module | Area | Status |
| :--- | :--- | :--- |
| Module 4 | Dataset Ingestion and Validation Engine | Completed |
| Module 5 | Automated Preprocessing Engine | Completed |
| Modules 1-3 | See overall architecture documentation | See `ml_pipeline_architecture.md` |
| Modules 6-8 | See overall architecture documentation | See `ml_pipeline_architecture.md` |

The table intentionally does not assign implementation details or completion claims to modules whose detailed requirements are not defined in this README. Refer to the architecture document for the complete project scope.

## 27. Design Principles

The Hospital Client follows these principles:
- **Privacy First:** Medical data remains localized to the client environment during the documented ingestion and preprocessing stages.
- **Non-Destructive Processing:** Original source files are not modified by the preprocessing workflow.
- **Clear Module Boundaries:** Each module has a defined responsibility and interface.
- **Leakage Prevention:** Training-derived preprocessing parameters are kept separate from validation and test information.
- **Patient-Level Safety:** Grouped splitting is supported where patient/group identifiers are available.
- **Scalability:** Lazy processing and incremental discovery reduce unnecessary memory and storage requirements.
- **Auditability:** Validation failures, quarantine decisions, manifests, and reports provide traceability.
- **Reproducibility:** Explicit configuration, deterministic splitting, and documented module contracts support repeatable execution.

## 28. Overall Data Flow

The documented Hospital Client flow can be summarized as:

```text
                    LOCAL HOSPITAL DATA
                            │
                            ▼
              ┌─────────────────────────┐
              │ Module 4                │
              │ Dataset Ingestion &     │
              │ Validation              │
              ├─────────────────────────┤
              │ • Read-only ingestion   │
              │ • Privacy / PHI checks  │
              │ • Format validation     │
              │ • Duplicate hashing     │
              │ • DatasetProfile        │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Module 5                │
              │ Automated Preprocessing │
              ├─────────────────────────┤
              │ • Quality validation    │
              │ • Quarantine            │
              │ • Splitting             │
              │ • Leakage prevention    │
              │ • Class balancing       │
              │ • Transformations       │
              │ • Lazy/materialized     │
              │ • Manifests & reports   │
              └────────────┬────────────┘
                           │
                           ▼
                 ML-READY ARTIFACTS
                           │
                           ▼
                 DOWNSTREAM TRAINING
                           │
                           ▼
             FEDERATED LEARNING PIPELINE
```

For the complete federated-learning architecture and the responsibilities of all project modules, see `ml_pipeline_architecture.md`.

## 29. Current Repository Readiness

The documented Hospital Client preprocessing foundation currently provides:

```text
Module 4
    Dataset ingestion and validation
    Dataset profiling
    Privacy checks
    Duplicate detection
             │
             ▼
Module 5
    Automated preprocessing
    Quality validation
    Quarantine
    Leakage prevention
    Patient/group-aware splitting
    Class balancing
    Lazy/materialized output
    Storage safety
    Reporting
    Testing
             │
             ▼
Downstream ML Training
```

Module 4 and Module 5 therefore establish the validated and preprocessing-controlled data boundary required before the downstream ML training stages.

