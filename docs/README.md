# Documentation Guide

This is the starting point for reading the project. It explains, in plain terms, what every
module does, which methods it actually uses, and where to read more. Everything below describes
the code as it is in this repository (checked against the source on 2026-10-09). Where something
is a heuristic, a default, mock data or not implemented, it says so.

> Research/engineering project. Model outputs are predictions, not clinical diagnoses. Federated
> learning keeps raw images on the hospital machine, but on its own it is not a privacy or legal
> guarantee.

---

## 1. Reading order

| If you want to understand... | Read |
|---|---|
| The whole platform in 10 minutes | this file, sections 2-4 |
| How a dataset becomes a trained model | this file, section 5, then the module docs it links |
| How the model and settings change with the machine | section 5.5 (Module 8) |
| How one training script trains every model | section 5.4 (Modules 6 + 7) |
| How federated rounds work | section 6, then `module9_readme.md` |
| How to install and run | `../README.md`, or section 9 here for the short version |
| Login, roles, accounts | `module1_readme.md` |
| Researcher/admin screens | `module2_readme.md` |
| Hospital screens | `module3_readme.md` |

Per-module documents (each starts with an "At a glance" summary, then the full details):

| Module | Document |
|---|---|
| 1 Authentication & roles | [module1_readme.md](module1_readme.md) |
| 2 Researcher/Admin desktop | [module2_readme.md](module2_readme.md) |
| 3 Hospital desktop | [module3_readme.md](module3_readme.md) |
| 4 Dataset inspection | [module4_readme.md](module4_readme.md) |
| 5 Preprocessing | [module5_readme.md](module5_readme.md) |
| 6 Model management | [module6_readme.md](module6_readme.md) |
| 7 Local training | [module7_readme.md](module7_readme.md) |
| 8 Resource-aware training | [module8_readme.md](module8_readme.md) |
| 9 Federated learning | [module9_readme.md](module9_readme.md) |
| 16 Local inference | [module16_readme.md](module16_readme.md) |

Older design documents at the repository root: `architecture.md` (platform and team),
`ml_pipeline_architecture.md` (ML pipeline design), `m9_implementation_plan.md` (Module 9 plan).
Where they disagree with the module docs, the module docs and the code are current.

---

## 2. Module status (all 21 modules)

**Implemented** = working code with automated tests. **Partial** = some of the module exists.
**Mock** = the screen exists but shows demo data with a "Mock data" banner. **Not implemented** = nothing yet.

| # | Module | Status | What exists |
|---|---|---|---|
| 1 | Authentication & roles | Implemented | Register, login, JWT, roles (admin / researcher / hospital_operator), admin approval, password change, audit log (Node + MongoDB) |
| 2 | Researcher/Admin desktop | Implemented (some screens mock) | Real: dashboard, training requests, monitoring, hospitals, federation jobs/rounds, global models, users, audit log, update validation. Mock: Experiments, Notifications, Messages |
| 3 | Hospital desktop | Implemented | Dataset, preprocessing, training, monitor, results, models, inference, federation status, settings. Operator messages are mock |
| 4 | Dataset inspection | Implemented | Image folders, CSV, XLS, XLSX; labels from class folders or a metadata CSV |
| 5 | Preprocessing | Implemented | Quality checks, quarantine, splits (stratified / grouped / existing), leakage checks, manifests |
| 6 | Model management | Implemented | 8 architectures, versioned safetensors checkpoints, integrity checks |
| 7 | Local training | Implemented | One PyTorch trainer for every architecture, metrics, plots, federation handoff |
| 8 | Resource-aware training | Implemented | Hardware detection, per-model memory/time estimates with a real dry run, recommendations, OOM retry |
| 9 | Federated learning | Implemented (custom protocol for the app + Flower simulation bench) | Jobs, join requests with admin approval (hospital sends its local eligibility summary), rounds, base-model distribution, update validation, weighted FedAvg, FedProx, round plan (total rounds, early stopping, automatic next round), global models, evaluation, promotion; `python -m federation.flower_sim` runs Flower with N simulated hospitals (IID / label-skew) |
| 10 | Differential privacy | Not implemented | |
| 11 | Secure communication | Not implemented | Local services use plain HTTP on localhost; no TLS/mTLS |
| 12 | Byzantine/malicious update detection | Not implemented | Module 9 has a hook that currently accepts every update |
| 13 | Monitoring & telemetry | Partial | Training progress, Module 8 hardware measurements, admin hospital telemetry |
| 14 | Database, audit, model storage | Partial | MongoDB (users, hospitals, requests, audit, notifications), SQLite for Module 9, model files on disk |
| 15 | Model versioning | Partial | Local versions (Module 6) and global model versions/promotion (Module 9) |
| 16 | Local inference | Implemented | Prediction on a new local image with a chosen model version |
| 17 | Mobile app | Not implemented | |
| 18 | Communication & notifications | Partial | Real: hospital notifications for round opened/closed (MongoDB + live Socket.io). Mock: messages |
| 19 | Testing & reliability | Partial | Unit/integration tests per module (see section 8) |
| 20 | Research experiments | Not implemented | Experiments screen is mock |
| 21 | Packaging & deployment | Not implemented | Run from source (section 9) |

---

## 3. How the parts fit together

```
 Admin / Researcher browser            Hospital browser (on the hospital machine)
          |                                   |                 |
          | REST + Socket.io                  | REST            | REST (dataset path, training choice)
          v                                   v                 v
   Backend API (Node, :5000) <---- verifies login ---- Local ML service (Python, 127.0.0.1:8765)
     |        |                                             |
     |        +-- spawns python federation/api_adapter.py   +-- runs python -m hospital_client.<module>
     v                     |                                     (Modules 4, 5, 8->7->6, 16)
  MongoDB                  v                                     reads the dataset locally
  (users, hospitals,   Module 9 (Python): SQLite federation_metadata.db
   requests, audit,                       + federation_artifacts/ (updates, base and global models)
   notifications)
```

- Raw images and datasets are only read on the hospital machine. The backend never receives them.
- A hospital sends the backend: aggregate training progress (epoch, loss, accuracy), and for a
  federated round, its model update (parameters + metadata, no images).
- **Limitation:** a hospital submits its update by sending the server the *folder path* of the
  update files, so today the hospital and the federation server must be on the same machine.

---

## 4. The hospital pipeline in one picture

```
dataset folder / CSV / Excel
   |
   v  Module 4  inspect        -> dataset_profile.json   (what is in the data: classes, splits, counts, issues)
   v  Module 5  preprocess     -> train/validation/test manifests + preprocessing report
   v  Module 8  recommend      -> 3 training options sized for THIS machine (model, batch size, epochs, precision)
   v  Module 7  train          -> best checkpoint (Module 6 format), metrics, plots, federation handoff
   v  Module 16 predict        -> prediction for a new image
   v  Module 9  (optional)     -> the handoff is submitted to a federation round
```

In the hospital app, steps 4 -> 5 -> 8 run automatically after you pick a dataset. Training starts
only when the operator picks one option.

All outputs go to `~/medfl_ml_work/<hospital_id>/` (change with `ML_WORK_DIR`):

```
datasets/<dataset_id>/   source.json, m4/ (profile), m5/ (manifests + report), recommendations.json
trained/<architecture>/  models/<model_id>/v<N>/ (checkpoint), <model_id>_run.json, _training_result.json, plots
federation_runs/<job>/   filtered manifests for a federated run
tmp/                     downloads in progress (e.g. a round's base model)
```

Each hospital id gets its own folder, so on a shared machine one hospital never sees another's files.

---

## 5. How each ML module works (methods actually used)

### 5.1 Module 4 - Dataset inspection

**Question it answers:** what is in this dataset? It never changes or copies the data.

- **Image folders:** walks every sub-folder. Supported: `.jpg .jpeg .png .bmp .tif .tiff .webp`.
  - Each image is opened with Pillow and verified. Zero-byte or unreadable files are counted as invalid.
  - Records width/height range, colour modes (`RGB`, `L`), formats.
  - **Duplicates:** SHA-256 hash of every valid file; equal hashes count as exact duplicates.
  - **Classes:** from folder names (`benign/`, `malignant/`). `train/`, `val/`, `validation/`,
    `test/` folders are recognised as existing splits, not classes.
  - **Labels from a metadata CSV (optional):** for datasets where all images are in one folder and
    a CSV says each image's class (e.g. ISIC). You choose the image-id column, the label column, an
    optional lesion/patient-id column, and a class mapping (rename values, merge several into one
    class, or exclude one). Unmatched or unlabelled images are counted, never guessed.
  - **Imbalance warning:** when the largest class has more than twice the images of the smallest.
- **CSV / XLS / XLSX:** column types (numeric, categorical, text), missing values, exact duplicate
  rows, constant columns, high-cardinality (ID-like) columns, candidate target columns. Module 4
  never picks the target column.
- **Output:** `dataset_profile.json` + `dataset_report.md`, with counts and names only, no pixels or rows.

Details: [module4_readme.md](module4_readme.md).

### 5.2 Module 5 - Preprocessing and splitting

**Question it answers:** how should this data be prepared for training? Input: the Module 4 profile.

- **Image quality check** (every image):
  - unreadable or zero-byte -> `REJECTED`;
  - optional **brightness** check (mean grey level below/above configurable thresholds) -> `LOW_QUALITY` (review);
  - optional **blur** check: variance of the Laplacian (a 3x3 edge filter computed with NumPy) below
    a configurable threshold -> `LOW_QUALITY`.
  - Brightness and blur thresholds are **off by default** (the app uses the defaults), so by
    default only unreadable files are excluded. Nothing is deleted: rejected/review images stay in
    the report (quarantine).
- **Labels:** from the folder path, or from the metadata CSV recorded by Module 4. With a CSV class
  mapping, the original value is kept as a `subtype` (e.g. `MEL` under `malignant`).
- **Splitting** (seed 42, ratios 80/10/10 train/validation/test):
  - If the dataset already has `train/`, `val/`, `test/` folders, they are **checked and kept**
    (duplicate images across splits and patient leakage are checked; an invalid split stops with an error).
  - **Only `train/` and `test/`:** the app carves a validation split out of `train/` (test untouched).
  - **No folders:** `StratifiedShuffleSplit` (keeps each class's proportion in every split). Falls
    back to a seeded random split only if a class has fewer than 2 images.
  - **Grouped (lesion/patient-level):** `GroupShuffleSplit` when real group ids exist (a metadata
    CSV group column or a `{relative_path: patient_id}` JSON map), so one patient/lesion is never in
    two splits. Group ids are never invented; without them the report says leakage could not be checked.
- **Class imbalance:** counts and inverse-frequency class weights are computed from the **training
  split only** (`n_samples / (n_classes * count)`) and stored in the manifest. Validation and test
  are never balanced. By default no rebalancing is applied.
- **Tabular data:** split first, then missing-value imputation (median for numbers, mode for
  categories) and optional encoding/scaling **fitted on the training split only**, then applied to
  validation/test (prevents leakage).
- **Augmentation** is not baked into the data; only the requested flags are recorded for Module 7
  to apply to the training split at training time. The app requests none.
- **Output modes:** `lazy` (default): manifests that point at the original files, no copies.
  `materialized`: resized/converted copies in a separate folder (after a free-disk-space check).
- **Accounting:** `discovered = accepted + rejected + review`, checked and shown as "Accounting: balanced".

Details: [module5_readme.md](module5_readme.md).

### 5.3 Module 6 - Model management

- **8 architectures** in one registry: ViT-B/16, EfficientNet-B4 (high-end), EfficientNet-B0,
  ResNet-50 (medium), ResNet-18, MobileNetV2 (low), MobileNetV3-Small, MobileViT-XXS (very low).
  Torchvision models, except MobileViT-XXS (implemented in the project).
- **Building a model for a task:** the architecture is created, then its final classification layer
  is **replaced** with a new layer that has one output per class of your dataset. For grayscale
  (`L`) data the first convolution is adapted to 1 channel.
- **Pretrained weights:** supported only from a local copy; the app trains **from scratch**
  (`pretrained=False`).
- **Checkpoints:** saved as `model.safetensors` (no pickle) + `metadata.json` under
  `models/<model_id>/v<N>/`. Versions are immutable (never overwritten). Every load verifies the
  SHA-256 hash and checks architecture, layer names and shapes strictly.

Details: [module6_readme.md](module6_readme.md).

### 5.4 Module 7 - Local training: one trainer for every model

There is **one** training loop (`hospital_client/training/trainer.py`) for all 8 architectures.
It never looks at which architecture it is training:

1. Module 8 (or the CLI) produces a `TrainingConfig`: architecture name, number of classes,
   input size, device, precision, batch size, epochs.
2. The trainer asks Module 6's registry to `build_model(config)`. The registry returns a standard
   PyTorch `nn.Module` with the right number of outputs. That is the only architecture-specific step.
3. The same loop then runs for any model: forward -> cross-entropy loss -> backward -> optimizer step.

**What the loop does:**
- **Data:** reads Module 5's manifests; each image is converted to the colour mode, resized to the
  architecture's input size (224x224; 380 for EfficientNet-B4; 256 for MobileViT-XXS) and turned
  into a tensor with values 0-1 (no mean/std normalisation). Training-only augmentations
  (horizontal/vertical flip, +/-15 degree rotation) are applied only if Module 5 requested them.
- **Defaults used by the app:** Adam, learning rate 0.001, no weight decay, no LR scheduler, no
  class weighting, no early stopping, seed 42. (Available but not used by the app: SGD, StepLR,
  cosine, reduce-on-plateau, early stopping, class weights from the manifest or computed.)
- **Mixed precision:** fp16 autocast with gradient scaling on CUDA when Module 8 selects it; fp32 on CPU.
- **Best checkpoint:** after each epoch the validation loss is computed; whenever it is the lowest
  so far, a new checkpoint version is saved. The final model is that best epoch, not the last one.
- **Metrics:** accuracy, macro precision/recall/F1, per-class precision/recall/F1/support,
  confusion matrix (scikit-learn). On the **test split** (only evaluated at the end, never used for
  choosing the checkpoint): the same metrics plus ROC curves, ROC-AUC (macro) and average precision,
  with confusion-matrix/ROC/PR plots.
- **Federated runs:** the round's base model is loaded and its checksum verified before the first
  training step; the run's class mapping is fixed by the job; the result is packaged as a
  `FederationHandoff` (parameters as `.npz` + metadata JSON).
- **Progress:** one line per epoch (`[epoch 3/10] train_loss=... val_acc=...`) that the app turns
  into live charts.

Details: [module7_readme.md](module7_readme.md).

### 5.5 Module 8 - How the model and settings change with the machine

Module 8 decides *which model and which settings* this particular machine can train, using real
measurements, then hands Module 7 a normal `TrainingConfig`.

1. **Detect hardware:** GPU name, total/free VRAM (`torch.cuda.mem_get_info`), GPU utilisation
   (`nvidia-smi`), CPU cores and load, RAM and free disk (`psutil`).
2. **Device:** CUDA if usable, otherwise CPU (for the whole machine, not per model).
3. **Memory budget:**
   - GPU: `(free VRAM - 400 MB driver overhead) x 0.80`
   - CPU: `available RAM x 0.70`
4. **For each of the 8 architectures:**
   - **Batch size:** try 64, 32, 16, 8, 4, 2, 1 (largest first). A memory estimate (parameters,
     Adam's extra state, activations per pixel) skips sizes that clearly do not fit. The first size
     that passes is then **actually tested**: one real forward + backward + optimizer step on this
     machine. If it runs out of memory or exceeds the budget, the next smaller size is tried. No
     size fits -> the model is marked **Unsafe** and cannot be chosen.
   - **Precision:** fp16 on CUDA, fp32 on CPU.
   - **Epochs:** the machine is put in a tier by its free VRAM (or RAM on CPU). The tier gives a
     time budget (20 / 30 / 45 / 60 minutes per 25,000 training images, scaled with dataset size).
     Epochs = how many fit in the budget, between 3 and 10 (default 10, max 40).
   - **DataLoader workers:** 0, 2 or 4, keeping 2 CPU cores free.
   - **Estimated time:** from the measured dry-run speed (or earlier real runs), shown as a range.
5. **Three options** (resource choices, not accuracy rankings):
   - **Recommended:** the heaviest model tier that still trains the full 10 epochs within the
     budget, and that the dataset is big enough for (ViT-B/16 and EfficientNet-B4 need at least
     50,000 training images when trained from scratch).
   - **Fast:** one tier lighter, or the same model with fewer epochs.
   - **High-Capacity:** one tier heavier, or more epochs; labelled "Not Recommended" if over budget.
   - A **manual** choice is also possible, but Unsafe models are refused.
6. **During training:** if CUDA runs out of memory, the run is retried with half the batch size,
   at most 2 times. Architecture and device are never changed silently.

So on a small GPU (e.g. RTX 3050, 4 GB) you typically get a light/medium model with a smaller
batch size and fp16. A larger GPU allows heavier models and bigger batches. A CPU-only machine gets
fp32, smaller models and fewer epochs. All thresholds are in
`hospital_client/resource_training/policy.py`. They are declared approximations that can be
recalibrated, not measured universal facts.

Details: [module8_readme.md](module8_readme.md).

### 5.6 Module 16 - Local inference

- The operator picks a specific trained model **and version**.
- The checkpoint is loaded through Module 6, so the integrity and shape checks apply.
- The new image is validated with Module 4's checks and preprocessed with the **same transform as
  Module 7's validation/test** (stored with the model).
- Inference runs locally (`torch.inference_mode`). Softmax gives class probabilities, and the
  predicted class comes from the model's stored class mapping.
- The image is never uploaded or written anywhere. Results are model predictions, not diagnoses.

Details: [module16_readme.md](module16_readme.md).

---

## 6. Federated learning (Module 9), short version

It is a **central** federated learning server (not peer-to-peer), implemented in this project
(`federation/`), not with Flower.

1. **Admin creates a job:** task (e.g. "Skin Cancer"), architecture, class mapping, minimum participants,
   and optionally a total number of rounds (the next round then opens by itself after each aggregation).
   **Hospitals ask to join:** each hospital runs the eligibility check on its own machine and sends the admin only
   the summary (eligible, classes covered, image counts, hardware tier). The admin approves or rejects; only
   approved hospitals can join the job's rounds.
2. **Admin opens a round:** a deadline and a minimum number of participants. The server prepares
   the round's **base model**:
   - round 1: one freshly initialised model shared by everyone (`SEED_<job>`);
   - later rounds: the job's latest global model.
   It stores the model's checksum in the round. Every active hospital is notified.
3. **An approved hospital joins:** its dataset must be tagged with the same task and contain the job's classes.
   - The local ML service downloads the base model.
   - Module 7 verifies its checksum and trains from it.
   - The update (parameters + metadata) is submitted to the round.
4. **The server validates each update:** the hospital is a participant; the architecture, classes
   and base model match the round; the parameter checksum is correct; no duplicate; before the deadline.
5. **Admin aggregates:** a confirmation lists the participating hospitals and their status. Closing
   before the deadline is allowed if the minimum number of accepted updates is met. Aggregation is
   **weighted FedAvg**: each hospital's parameters are weighted by its number of training images.
   Integer buffers are copied, and NaN/Inf fails the aggregation. Hospitals that did not join are
   notified that the round is closed.
6. **Result:** a new global model version (status `EVALUATION_PENDING`). The admin can evaluate it
   and promote it. The next round starts from it.

Not implemented: differential privacy (Module 10), TLS (Module 11), malicious-update detection
(Module 12, hook accepts everything). Details: [module9_readme.md](module9_readme.md).

---

## 7. Known limitations (current)

- Hospital and federation server must share a machine (updates are submitted as a folder path).
- No TLS, no differential privacy, no Byzantine defence.
- Module 5's grouped split keeps patients/lesions together but does not stratify by class. Check
  the per-split class counts in the report.
- Images are not mean/std-normalised, and models train from scratch. Fine for experiments, but
  accuracy on small datasets will be lower than with pretrained weights.
- Several admin/researcher screens are mock (marked with a banner).

---

## 8. Tests

| Suite | Command | Last result |
|---|---|---|
| Hospital ML pipeline (Modules 4-8, 16, local ML service) | `python -m pytest hospital_client -q` | 828 passed, 1 skipped (~16 min) |
| Module 9 federation | `python -m pytest federation/tests -q` | 31 passed |
| Backend (Modules 1-3, 9 routes, notifications) | `cd backend; npx jest` | 159 passed, 2 failing (stale: they expect hospitals to be refused the job list) |
| Frontend build | `cd frontend; npx vite build` | builds |

---

## 9. Run the project (short version)

One-time setup is in `../README.md` section 3. Then, from the repository root, in three terminals:

```powershell
cd backend; npm start                    # backend API      http://localhost:5000
python -m hospital_client.local_api      # local ML service http://127.0.0.1:8765 (from the repo root)
cd frontend; npm run dev                 # web app          http://localhost:5173
```

Open http://localhost:5173 and sign in.
