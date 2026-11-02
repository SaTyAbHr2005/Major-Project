# Module 3: Hospital Desktop Application

**Project:** Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging  
**Owner:** Member 4 (Frontend + Desktop + Mobile + Communication)  
**Location:** `frontend/` (consolidated researcher + hospital app; hospital screens are `frontend/src/views/Hospital*View.jsx`)  
**Status:** Implemented; ML screens use the real pipeline, operator messages are mock  
**Last Updated:** 2026-11-02

> **Integration update (2026-09-27).** The separate `frontend/hospital-desktop/` and
> `frontend/researcher-desktop/` apps were removed; `frontend/` is the single app. The hospital
> ML screens (Modules 4, 5, 6, 7, 8, 16) no longer use mock services: they call the hospital-local
> ML service (`python -m hospital_client.local_api`, 127.0.0.1:8765), which runs the real module
> CLIs on this machine. All numbers shown on those screens come from real pipeline output. See
> `ml_pipeline_architecture.md` → *Hospital Desktop Integration*. Federation (Module 9), including DP jobs
> (Module 10) and TLS (Module 11), now runs through the Federation Status screen. Module 15 approval is still pending.

---

## Status & Implementation Checklist

- [x] Hospital Operator Dashboard with measured workstation hardware (Module 8 `detect`)
- [x] Module 4: locate a dataset (native folder picker or path) and view the real DatasetProfile
- [x] Module 5: automatic preprocessing with accounting, split sizes and leakage checks; optional re-run with a patient/group-id map
- [x] Module 8: up to 3 hardware- and data-aware recommendations, a same-tier alternative for each, a manual configuration with a load meter, and all 8 model assessments; the operator must choose one explicitly
- [x] Module 7: launch training, live per-epoch loss/accuracy curves, log stream, terminate
- [x] Module 7: results with test metrics, per-class table, confusion matrix, ROC/PR plot images, resource statistics
- [x] Module 9: join requests, automatic training during Flower rounds, step reports, update submission and retry
- [x] Module 6: local model registry (versions, class mapping, SHA-256); Module 15 approval pending
- [x] Module 16: local inference on an uploaded image (temporary copy deleted after prediction)
- [x] Optional link of a local run to a backend training request (aggregate progress only)
- [x] Module 18: real round notifications (operator messages are still mock)
- [x] Module 1: Authentication & Strict Hospital Isolation Adapter (`authService.js`)
- [x] Workstation Settings (UI)

---

## 1. Module Purpose & Core Architecture

Module 3 is the hospital-side part of the web app, run in a browser on the hospital machine; desktop packaging is Module 21. It is the workstation interface for hospital operators taking part in federated deep learning.

### Cardinal Privacy Invariant
> **RAW MEDICAL DATA NEVER LEAVES THE HOSPITAL.**  
> Medical images and patient data remain local. Only model updates, required metadata and authorized telemetry are transmitted to the central server.

```text
                               +------------------------------------------+
                               |     Module 3: Hospital Desktop App       |
                               | (React 18 + Vite + CSS Design System)    |
                               +--------------------+---------------------+
                                                    |
         +------------------------------------------+------------------------------------------+
         |                                          |                                          |
         v                                          v                                          v
+------------------+                      +-------------------+                      +-------------------+
|  Data & Pipeline |                      | Resource & Train  |                      | Clinical Support  |
+------------------+                      +-------------------+                      +-------------------+
| • datasetService |                      | • resourceService |                      | • modelService    |
|   (Module 4)     |                      |   (Module 8)      |                      |   (Module 6)      |
| • preprocess-    |                      | • trainingService |                      | • inferenceService|
|   ingService     |                      |   (Module 7)      |                      |   (Module 16)     |
|   (Module 5)     |                      | • Fed Handoff     |                      | • communication-  |
+--------+---------+                      |   (Module 9)      |                      |   Service (M18)   |
         |                                +---------+---------+                      +---------+---------+
         |                                          |                                          |
         +------------------------------------------+------------------------------------------+
                                                    |
                                                    v
                               +------------------------------------------+
                               |   Backend & Consortium Infrastructure    |
                               |  • Module 1: Auth & Hospital Isolation   |
                               |    (HOSP_000001, ROLES.HOSPITAL_OPERATOR)|
                               |  • Module 9: Central Aggregator          |
                               +------------------------------------------+
```

The ML services (`datasetService`, `preprocessingService`, `resourceService`, `trainingService`,
`modelService`, `inferenceService`) talk to the **hospital-local ML service** through
`services/mlClient.js`, not to the backend: raw images and datasets never go to the backend.
Shared ML state (active dataset, pipeline progress, hardware, training job, training runs,
local models) lives in the Zustand store `stores/mlStore.js`. It is reset whenever a user signs
in or out, and the running training job is resumed after a page reload.

---

## 2. Responsibilities Matrix

| Responsibility | Handled by Module 3 | Related Modules |
|---|:---:|:---:|
| Clinical operator login & hospital session isolation | ✅ | Module 1 (Auth & Isolation) |
| Local dataset selection & inspection results (image folders, CSV/XLS/XLSX) | ✅ | Module 4 (Dataset Ingestion) |
| Preprocessing results, split accounting & leakage checks | ✅ | Module 5 (Preprocessing) |
| Hardware inspection & 3-option training recommendation | ✅ | Module 8 (Resource-Aware Engine) |
| Local deep learning training monitor & loss/accuracy curves | ✅ | Module 7 (Local Training) |
| Federation participation and update submission (`FederationHandoff` built by Module 7) | ✅ | Module 9 (FL Engine) |
| Local model registry view | ✅ | Module 6 (Model Management) |
| Local inference entry point | ✅ | Module 16 (Local Inference) |
| Round notifications (operator messaging is mock) | ✅ | Module 18 (Communication) |
| Raw image transmission outside hospital network | ❌ *(Permanently blocked)* | Core Architectural Invariant |
| Central model aggregation across multiple hospitals | ❌ | Handled centrally in Module 9 |

---

## 3. Hospital Isolation & Security Boundaries

Module 3 enforces the strict security rules of Module 1:

1. **Role Enforcement:** Only `hospital_operator` accounts see these screens, and the local ML service accepts only those accounts. Admin and Researcher are separate roles with their own screens (Module 2).
2. **Hospital Isolation:** Every action is scoped to the operator's assigned `hospital_id` (e.g. `HOSP_000001` - *St. Jude Clinical AI Node*). Operators cannot view or query data from any other hospital node.
3. **Raw data stays local:** Network payloads are limited to model updates, sample counts, the local eligibility summary and aggregate training progress.

---

## 4. Key Workflows & Screen Breakdown

### 1. Hospital Operator Dashboard (`HospitalDashboardView.jsx`)
- Hospital node banner and local ML service status (online/offline, with the start command when offline).
- Measured hardware: GPU name and compute capability, free/total VRAM, free/total RAM, CPU cores/threads, free storage (Module 8 `detect`).
- Active dataset summary, current training state, and the most recent local training runs read from disk.

### 2. Module 4: Dataset Ingestion & Inspection (`HospitalDatasetView.jsx`)
- The operator only **locates** the dataset: a native folder picker ("Browse...") on this workstation or a typed path. Inspection, preprocessing and recommendations then run automatically, with a step tracker and the live command log.
- Real DatasetProfile: total/valid/invalid samples, exact duplicates, class distribution, detected splits, image dimension ranges, color modes and formats, tabular/missing-data statistics, warnings/errors, and the generated `dataset_report.md`.
- Previously inspected datasets can be re-selected.

### 3. Module 5: Automated Preprocessing Engine (`HospitalPreprocessingView.jsx`)
- Shows the real preprocessing report: discovered/accepted/rejected/review accounting, split sizes, split strategy, leakage checks, and `preprocessing_report.md`.
- Optional re-run with CLI options: `lazy`/`materialized` mode, invalid-split policy, patient/group-id map (enables patient-level splits), tabular target column.

### 4. Module 8: Resource-Aware Training Recommendations (`HospitalStartTrainingView.jsx`)
- Up to three options generated for this machine and dataset: **Recommended**, **High-Capacity** (labelled **Not Recommended** when it exceeds the time budget or the dataset is too small for that model tier), and **Fast**. Each shows device, precision, batch size, epochs, workers, estimated time range, measured memory, rationale and trade-off.
- A table of all 8 architectures with tier, parameters, status (e.g. EfficientNet-B0 is always listed), batch size, epochs, time and memory.
- Each option also shows a **same-tier alternative** (e.g. Recommended ResNet-50 / alternative EfficientNet-B0).
- **Manual Configuration** panel: any feasible model, a batch size up to the largest one Module 8 measured as safe for it, and 1-40 epochs, with a live **load meter** (GPU memory or RAM, training time vs Module 8's time budget, CPU used by data loading; each bar says whether it is measured or estimated).
- **Nothing is pre-selected or started automatically**: the operator must pick one option and confirm in a modal.
- Optional link to a backend training request (join with "Participate"); only aggregate progress is reported.

### 5. Module 7: Local Training Monitor (`HospitalTrainingMonitorView.jsx`)
- Job status, epoch progress, elapsed time and an ETA derived from the measured average epoch time.
- Live per-epoch train/validation loss and accuracy curves (parsed from Module 7's per-epoch output lines) and the full execution log.
- Terminate button (stops the whole local training process tree).

### 6. Module 7 & 9: Training Result & Federation Handoff (`HospitalTrainingResultView.jsx`)
- Validation and held-out test accuracy/loss/F1, macro ROC-AUC and average precision, per-class metrics and the confusion matrix from `<model_id>_training_result.json`.
- The evaluation plots generated by Module 7 (confusion matrix, ROC and precision-recall curves).
- Module 8 resource statistics (peak VRAM/RAM, CPU, GPU utilization when available, throughput, estimated vs actual time, OOM events, batch adaptations).
- Module 9 card: the FederationHandoff prepared locally by Module 7 (no images), which federated runs submit to the round.
- Run history table and a raw-JSON audit inspector.

### 7. Module 6: Local Model Registry (`HospitalModelsView.jsx`)
- Every trained version in the Module 6 store: architecture, version, parameters, size, input spec, class mapping, training settings and artifact SHA-256 (verified on load). Module 15 approval is shown as pending.

### 8. Module 16: Local Inference (`HospitalInferenceView.jsx`)
- Select a model and an explicit version, a device (auto/CUDA/CPU) and a local image; the image is sent only to 127.0.0.1, processed with the model's own preprocessing spec, and its temporary copy is deleted after the prediction.
- Predicted class, uncalibrated softmax confidence, full class probability distribution, timing, warnings, and the "model prediction, not a clinical diagnosis" notice.
### 9. Module 9: Federation Status (`HospitalFederationView.jsx`)
- Eligibility of the selected dataset for each federation job, the join request sent to the Admin (local eligibility summary only), the participate flow (themed confirm modal, then the Training Monitor), and this hospital's submitted updates with their validation status.
- In Flower jobs, approved hospitals train automatically while the app is open and signed in, and the screen shows the current step. DP jobs apply the job's noise settings automatically.
- A promoted global model cannot be downloaded here yet (Modules 15 and 16).
- Before a federated run starts, the local ML API downloads the round's base model from the backend into `trained/<architecture>/models/<base_model_id>/v<version>` (once per base model); Module 7 then refuses to train if its parameter checksum differs from the round's.
- **Retry Submission** for a finished federated run whose update never reached Module 9: the local ML API (`POST /api/ml/federation/submit/<model_id>`) resubmits the run's handoff folder to the job and round saved in its `<model_id>_run.json`. Runs started before the round id was saved cannot be resubmitted (409).

### 10. Settings (`HospitalSettingsView.jsx`)
- Hospital id and name (from `/auth/me`), the hardware Module 8 measured (GPU/VRAM, CPU, RAM, disk) and password change. There are no manual compute settings: Module 8 sizes every run from the measurements.

### 11. Module 18: Communications & Notifications (`HospitalCommunicationView.jsx`)
- **Notifications are real**: `GET /api/v1/notifications` returns this hospital's notifications (stored in MongoDB, `Notification` model), and new ones arrive live on the hospital's Socket.io room `hospital_<hospital_id>`. The federation server sends `ROUND_OPEN` when a round opens and `ROUND_CLOSED` to hospitals that did not join a round when the admin aggregates it. Opening the tab marks them read (`POST /api/v1/notifications/read`); the bell shows the unread count.
- **Operator messages are mock data** (no messaging API yet) and carry a "Mock data" banner.

---

## 5. Verification & Automated Test Suite

The mock-service unit tests of the old `frontend/hospital-desktop/` app (`node --test`) were removed
together with that app and its mock services. The integrated behaviour is covered by:

```text
python -m pytest hospital_client/local_api -q                 # local ML service: auth, origin checks,
                                                              # input validation, real M4->M5 run,
                                                              # job cancel, temp-file cleanup
python -m pytest hospital_client -q                           # full hospital suite (865 passed, 1 skipped, 2026-10-23)
cd backend && npm test                                        # includes tests/api/training_progress.test.js (188 passed, 2026-10-23)
cd frontend && npx vite build                                 # frontend compiles
```

A browser end-to-end run (2026-09-27, synthetic images, in-memory MongoDB) covered: hospital login,
locate dataset -> automatic M4/M5/M8, explicit recommendation choice, training with live curves,
terminate, results with plots, local inference, backend progress for a linked training request, and
every hospital and researcher screen loading without console errors.

---

## 6. How to Run Locally

### Prerequisites
- Node.js >= 20.19, npm >= 9 (full setup: root `README.md`)
- Python environment with `pip install -r requirements.txt`
- `backend/.env` and `frontend/.env` (see `backend/.env.example`; frontend needs `VITE_API_URL` and `VITE_ML_API_URL`)

### Start the three processes
```powershell
cd backend;  npm install; npm start            # REST API on http://localhost:5000/api/v1
python -m hospital_client.local_api            # local ML service on http://127.0.0.1:8765 (repo root)
cd frontend; npm install; npm run dev          # app on http://localhost:5173
```

Sign in with a hospital operator account from the backend database. The local ML service verifies
the session with the backend (`GET /auth/me`) and only accepts `hospital_operator` accounts.
Generated ML outputs go to `ML_WORK_DIR` (default `~/medfl_ml_work`), never into the dataset folder.