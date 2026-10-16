# Module 9 — Federated Learning and Global Model Management

## 1. Purpose and Scope
Module 9 (M9) is the project's **central** federated learning server/orchestrator. It is not peer-to-peer. A job runs either with manual rounds (the admin opens and aggregates each round, sections 3-22) or with Flower (one Flower server per job runs every planned round automatically, section 25); Flower is also used for the simulation bench (`federation/flower_sim.py`, section 24.3). It connects the hospital machines (through the local ML service) with the Consortium Admin (Node.js/React + Python API adapter). M9 manages federation jobs, round lifecycles, checksum-verified updates, weighted aggregation, and versioned Global Models.

> Updated 2026-10-09. Changes since the original write-up:
> - rounds are opened from the admin UI;
> - the server prepares each round's base model and hospitals download it;
> - round deadlines are validated;
> - early aggregation goes through a confirmation and progress dialog;
> - hospitals are notified when a round opens or closes.
>
> See sections 3, 5, 7, 14 and 22. Plain-language overview: `docs/README.md` section 6.
>
> Updated 2026-10-14: round plan (total rounds, early stopping, automatic next round), FedProx local term,
> Flower simulation bench (section 24); join requests with admin approval (section 6); evaluation stops
> when an evaluation image cannot be processed (section 15).

> Updated 2026-10-16: evaluation progress on the admin screen and the held-out evaluation set (section 15);
> the first real 3-round application run, evaluated (section 18); the Flower bench run on the real malaria data
> (section 24.4), with BatchNorm calibration after aggregation (section 24.3); known limitations (section 20).
> Flower in the application: Flower jobs, the admin's live Flower panel and automatic hospital training (section 25).

## 2. Architectural Overview
The architecture separates administrative orchestration from local training:
- **Consortium Admin (Central)**: Creates federation jobs/rounds, prescribes the canonical base model provenance, validates incoming hospital updates, executes aggregation, and holds the resulting Global Model artifact.
- **Hospital Client (Local)**: Verifies eligibility, obtains the prescribed canonical base model from its local model store, verifies its checksum, performs local training on private data, and submits a `FederationHandoff` payload.

## 3. Project-Specific Centralized Federation Aggregation Protocol
This project utilizes a **Project-Specific Centralized Federation Aggregation Protocol**. The server does not push models to hospitals: a hospital fetches the round's base model when it joins, and it trains when its operator starts training. With a round plan (section 24.1) the server side repeats on its own: after each aggregation the next round opens from the new global model until the planned total or the stopping rule is reached. Without a plan, the admin opens every round.

**The Workflow:**
1. **Admin** creates a federation job (optionally with a round plan).
2. **Hospitals** run the eligibility check locally and **request to join**; the **admin approves or rejects** each request (section 6).
3. **Admin** opens a round (deadline + minimum participants), or the round plan opens it. M9 prepares the round's **canonical base model**:
   - round 1: one freshly initialised seed;
   - later rounds: the job's latest global model.
4. **Approved hospitals** join the round. Each one downloads the base model from the server.
5. **Hospitals** train locally (M7) and produce checkpoint artifacts.
6. **M9 (Central)** receives and strictly validates these updates.
7. **M9** aggregates the validated parameter arrays.
8. A single **Global Model** is created centrally.
9. **Admin** evaluates the model.
10. **Admin** makes a promotion decision.

**Key Boundary:** The Global Model is centrally controlled. Hospitals receive a global model's weights only as the **base model of a later round they join** (authenticated download). There is no general push to all hospitals. Researchers have no access to global model weights.

## 4. Core M9 Data Model
All federation state tracking is persisted in the SQLite database (`federation_metadata.db`) using the following core entities:
- **`FederationJob`**: Defines the overarching task, architecture, class mapping, expected participants, and aggregation rules.
- **`FederationRound`**: A specific training iteration containing the canonical `base_model_id`, `base_model_version`, and `base_model_checksum`.
- **`FederationUpdate`**: An individual hospital's submission containing the artifact paths, metadata, and validation status.
- **`GlobalModelVersion`**: The finalized, aggregated model resulting from a successful round.

## 5. Federation Round Lifecycle
A round strictly follows this state machine:
`CREATED` → `OPEN` → `RECEIVING` → `READY_FOR_AGGREGATION` → `AGGREGATING` → `GLOBAL_MODEL_CREATED`

**Opening a round (adapter `create_round`)**:
- The deadline is required. It must parse as an ISO 8601 date-time with a timezone, lie in the future, and is stored in UTC. An empty deadline used to make every update count as late.
- `minimum_participants` defaults to the job's minimum.
- `expected_participants` may start empty. Hospitals are added when they join (`POST /federation/jobs/:job_id/rounds/:round_id/participate`).
- Unless a base model is passed explicitly, `federation/base_models.py` prepares it (section 7). The round opens immediately (`OPEN`).

**Admin Early-Close Mechanism (`force_close=True`)**:
- Permitted *only* through the authorized Admin aggregation path.
- Bypasses the future-deadline guard to force immediate evaluation of current submissions.
- **Does NOT bypass**: Minimum participant constraints, participant authorization, validation (architecture, schema, canonical provenance, duplicates), or standard expiration rules for submissions.
- Hospital submission paths are structurally isolated from this privilege.

**Deleting and archiving (added 2026-10-16)**. The admin's Delete removes a job permanently only when it has no
history: every round is still `CREATED`, `OPEN` or `RECEIVING` and no update was accepted or used in an aggregation.
Otherwise the job is **archived**: rounds, updates, global models and the audit trail are kept, because global models
record the job and round they came from. An archived job is read-only. Its plan reports it finished ("This job is
archived (read-only)."), so no round can be opened by the admin, automatically after an aggregation, or by a Flower
server, Start federation is refused, hospitals cannot request to join, and aggregation never changes its status back
to `COMPLETED`. Its global models can still be evaluated and promoted. The admin screen lists archived jobs in a
separate, closed "Archived jobs" section without action buttons. Rounds follow the same rule: Delete Round removes a
round without history and cancels one with history. Test: `federation/tests/test_round_plan.py`
(`test_an_archived_job_is_read_only`).

## 6. Hospital Participation and Authorization
Hospital identity relies on authenticated JWT/session context, not on untrusted client inputs.
- The Node.js API obtains `req.user.hospital_id` directly from the authenticated session.
- Node.js explicitly passes this verified identity to the Python M9 backend.
- M9 records and validates participant authorization against its own `expected_participants` mechanism defined in the Federation Job.

**Join requests (added 2026-10-14, `federation/participation.py`).** The server never sees a hospital's data, so it
cannot judge eligibility itself:
1. The hospital app runs the eligibility check on its own machine (`hospital_client/federation/eligibility.py`:
   task match, class coverage, image counts, hardware tier).
2. The hospital sends **Request to join** (`POST /api/v1/federation/jobs/:job_id/participation-request`) with only
   that summary. The server keeps only aggregate fields (booleans, counts, class names, short reasons; bounded
   sizes); anything else, such as paths, is dropped. An ineligible summary is refused.
3. The admin sees each request with the summary on the job card and approves or rejects it
   (`POST /api/v1/federation/jobs/:job_id/participants/:hospital_id/decision`). The hospital is notified.
4. Only **approved** hospitals can join the job's rounds (`/participate` returns 403 otherwise). Approval adds the
   hospital to the job's `expected_participants`, which update validation checks; rejection removes it, and drops it
   from rounds it joined but has not sent an update to.
- Hospitals that were in `expected_participants` before join requests existed count as approved.
- A hospital sees only its own request (`my_participation` in the job list), never other hospitals' summaries.
- The summary is reported by the hospital and **cannot be verified** by the server. It is decision support for the
  admin, like the sample count the update reports today; detecting dishonest hospitals belongs to Module 12.
- **Note**: The legacy M1-M3 MongoDB `TrainingRequest` records are out of scope for M9 and are not used for final M9 federation authorization.

## 7. Canonical Base-Model Provenance
M9 enforces canonical model provenance across all participating nodes so that each hospital begins local training from the same prescribed model parameters.

```text
Canonical Model
    ↓
FederationRound
    ↓
base_model_id / base_model_version / base_model_checksum
    ↓
Hospital obtains/verifies exact canonical artifact
    ↓
M7 Training
    ↓
FederationHandoff
```

**Checkpoint Identity vs. Canonical Base Identity:**
A hospital's resulting local checkpoint has its own unique identity (e.g., generated at training time), but its *provenance* must exactly match the round.
- **Hospital A**: `checkpoint_model_id = <Local_A>`, `base_model_id = CANONICAL_SEED_MOBILENET_V3`
- **Hospital B**: `checkpoint_model_id = <Local_B>`, `base_model_id = CANONICAL_SEED_MOBILENET_V3`

Before training, the hospital client recomputes the checksum of the downloaded base model and compares it with the round's prescribed checksum. Training **fails closed** if the artifact is missing, altered, or fails checksum validation.

**How the base model reaches the hospitals:**
1. `federation/base_models.py` `prepare_base_model()` builds the round's base model as a Module 6 checkpoint in `federation_artifacts/base_models/<id>/v1/`:
   - round 1: a new model `SEED_<job_id>` (draft, untrained);
   - later rounds: the job's latest global model's parameters, loaded into the architecture.
2. The checksum is computed exactly as Module 7 verifies it (SHA-256 over the loaded parameters). An existing base model is reused, not rebuilt.
3. Hospitals download it through `GET /api/v1/federation/rounds/:round_id/base-model/:file` (hospital_operator or admin). Only the two checkpoint files are served (`model.safetensors`, `metadata.json`).
4. The local ML service (`_ensure_base_model` in `hospital_client/local_api/server.py`) downloads it once into `trained/<architecture>/models/<id>/v1/`, using the operator's own token and the configured backend URL.
5. Module 7 then verifies the checksum before the first training step.

## 8. FederationHandoff Artifact Format
M9 separates hospital updates from the final M9 aggregated global models. However, both natively use the highly optimized `FederationHandoff` format.
- `handoff_parameters.npz`: Compressed NumPy format containing the raw model state tensors.
- `handoff_metadata.json`: Structural definitions, parameter shapes/dtypes, model metadata, and checksum/provenance information.

## 9. M9 Update Validation
When M9 receives an update, the validation engine strictly verifies:
- Authenticated participant identity is expected and authorized.
- Architecture, task, and schema match the job perfectly.
- **Canonical Provenance**: The hospital's submitted `base_model_checksum` and `base_model_id` precisely match the round's canonical requirements.
- **Artifact / Parameter Integrity**: M9 verifies the submitted artifact and its declared parameter integrity according to the `FederationHandoff` checksum metadata before accepting the update.

## 10. Aggregation Engine
The Aggregator merges `ACCEPTED` updates into a new global topology.

- **Floating-Point State**: Parameter tensors (e.g., weights/biases) undergo a weighted average based on `num_train_samples`.
- **Non-Floating State**: PyTorch structural parameters (like integer `num_batches_tracked`) are mathematically invalid for averaging. M9 implements an explicit policy: it copies the corresponding non-floating value from the first accepted participant.
- **NaN/Inf Protection**: The aggregation engine checks floating-point parameter arrays for `NaN` or `Inf` values and fails the aggregation rather than creating a Global Model from mathematically invalid parameters.

*Note*: Only updates reaching `ACCEPTED` are evaluated. Once utilized, their status transitions to `USED_IN_AGGREGATION`.

## 11. Global Model Creation and Persistence
Following successful aggregation, M9 instantiates a new `GlobalModelVersion` record and natively stores the resulting artifact as a `FederationHandoff` under `federation_artifacts/global_models/` directory.

## 12. Dynamic Participating-Hospital Provenance
M9 guarantees UI integrity by dynamically calculating historical provenance rather than permanently persisting stale columns.
```text
Global Model
    ↓
source round_id
    ↓
federation_updates (status == USED_IN_AGGREGATION)
    ↓
participant_id
    ↓
Admin API / React UI
```
The implementation dynamically derives participant lists for arbitrary participant counts (subject to resource constraints) straight from the database history, ensuring no hospitals are hardcoded in production logic.

## 13. SQLite Persistence Hardening
A critical persistence integrity fix was applied during M9 development.
- **Issue**: A migrated SQLite database had physical column drifting because `ALTER TABLE ADD COLUMN` appended metadata after timestamps, misaligning positional `INSERT ... VALUES (...)` queries.
- **Hardening**: Refactored the core `storage.py` schema handling to use **explicit named-column INSERTs** and **sqlite3.Row/named-field reads**.
- **Impact**: Completely isolates M9 persistence from future schema drift without requiring dangerous record deletion or weight modifications.

## 14. Admin API and React Integration
The central platform exposes secure Node.js controllers proxying requests to the Python `api_adapter.py`. 
- **Admin Roles**: Exclusively hold the authority to view global models, trigger rounds, execute aggregation, and force early close.
- **React UI**: Consumes dynamically derived provenance to accurately display the exact hospitals responsible for the specific model's aggregation.
- **Federation Jobs screen** (`frontend/src/views/FederatedTrainingView.jsx`):
  - create jobs;
  - **Open round N** (deadline, minimum participants; disabled while another round of the job is in progress);
  - the participant list with hospital names (admins get `hospital_names` with `GET /federation/jobs`).
- **Aggregate Round** (Federation Jobs and Admin Overview, `components/federation/AggregateRoundDialog.jsx`):
  1. A confirmation lists every participating hospital (name, id, update status) and warns when the deadline has not passed.
  2. Confirming sends `force_close: true`. Module 9 still requires the minimum accepted updates.
  3. The dialog shows the aggregation in progress, then the new global model, the round status and which hospitals were notified.
- **Hospital access:** `hospital_operator` may list jobs, join a round, submit its own update and download a round's base model. Everything else is admin-only.
- **Federation Models screen** (`frontend/src/views/AdminFederationModelsView.jsx`): evaluate a version (with live progress, section 15), compare it with `MAIN`, promote it.

## 15. Admin Evaluation Bridge to M6/M16
Global models are stored in M9 `FederationHandoff` format. To test global models, Admin evaluation executes a temporary structural bridge:
```text
M9 Global Artifact (.npz)
    ↓
FederationHandoff.load()
    ↓
PyTorch architecture reconstruction (set_parameters)
    ↓
Temporary M6 ModelStore (admin_m6_store/)
    ↓
model.pt + metadata.json
    ↓
M16 LocalInferenceEngine
```
This isolates the permanent M9 federation format from the evaluation mechanics of M16.

If any evaluation image cannot be processed, the evaluation **stops with an error** listing how many images failed
and the first few file names, and no metrics are recorded. (Before 2026-10-14 failed images were skipped silently,
so metrics could describe only part of the evaluation set.)

**Evaluation set.** The evaluator reads `evaluation_datasets/<task>/<dataset_version>/<class name>/<image>` relative
to the repository root (git-ignored; never committed). `dataset_version` defaults to `v1`; it is the version of the
evaluation set, not of the global model, so every global model of a task is scored on the same images and versions
can be compared. Class folder names must match the job's `class_mapping` labels.

`evaluation_datasets/Malaria/v1` (built 2026-10-15) holds 2,333 NIH malaria cell images (1,168 Parasitized,
1,165 Uninfected): every image that neither hospital of `JOB_MAL_MNV2_CLS2_1` used for training or validation in
any of its 3 rounds. Images were matched to the hospitals' train/validation manifests by cell name (including the
Roboflow JPG re-exports one hospital trained on) and by SHA-256; none matched. Classes were not re-balanced.
`eval_manifest.json` in that folder records the rule, the manifests checked and a SHA-256 per image (the evaluator
skips it). Limitation: the hold-out is per image, not per patient; every one of the 201 patients also has other
cells in the hospitals' training data, so the scores are likely higher than on new patients (section 20).

**Progress.** An evaluation of 2,333 images takes about 4 minutes on CPU. While it runs, the evaluator writes
`federation_artifacts/evaluation_progress/<global_model_id>.json` (`status` preparing/running/completed/failed,
`done`, `total`, `started_at`, `images_started_at`, `updated_at`, `error`) every 25 images. The backend serves it at
`GET /api/v1/federation/models/:global_model_id/evaluation-progress` (admin only; ids are checked so the path
cannot leave that folder; `progress: null` when none exists). The Federation Models screen polls it every 1.5 s
while its evaluate request is open and shows the stage, images checked, elapsed time and estimated time left; the
version's badge reads `EVALUATION_IN_PROGRESS` meanwhile. Writing progress is best effort: on Windows the file can
be locked while the backend reads it, so a write is retried briefly and otherwise skipped, and it can never stop
the evaluation.

## 16. Global Model Lifecycle
Upon successful creation, an aggregated Global Model assumes the status:
`EVALUATION_PENDING`

**There is no automatic promotion to MAIN.**
The Global Model is created with `EVALUATION_PENDING`. The subsequent lifecycle is an administrative workflow rather than a single database state machine:

`EVALUATION_PENDING`
→ Admin evaluation
→ comparison with the current `MAIN` model within the same model track, if one exists
→ manual Admin promotion decision
→ Admin promotion decision → promoted to `MAIN` within the same model track, or retained as a non-MAIN candidate.

## 17. Model Track / Schema Identity
Global models are explicitly tracked by a strict schema identity. A track requires mathematical alignment of:
- Disease / Task
- Architecture
- Number of Classes
- Canonical Class Mapping (Schema Hash)

*Example*: `Malaria + MobileNetV3-Small + 2 Classes` operates entirely isolated from `Malaria + ResNet18 + 2 Classes` or `Malaria + MobileNetV3-Small + 5 Classes`.

## 18. End-to-End Verification
The architecture was verified utilizing a complete, canonical-provenance end-to-end pass (`run_final_e2e.py`).

**Verified Execution Record:**
- **Job**: `JOB_MAL_MNV3S_CANONICAL_E2E_3`
- **Round**: `ROUND_MAL_MNV3S_CANONICAL_E2E_R3`
- **Canonical Model**: `CANONICAL_SEED_MOBILENET_V3`
- **Participants**: `HOSP_868720`, `HOSP_123456`
- **Participant Result**: Both `ACCEPTED` and `USED_IN_AGGREGATION`
- **Global Model ID**: `GM-MALARIA-MOBILENET_V3_SMALL-V1` (Created successfully via `weighted_fedavg`)

*(Note: Hospital IDs are historical verification inputs, not hardcoded logic.)*

**Application run with a round plan (2026-10-15).** `JOB_MAL_MNV2_CLS2_1`: Malaria, MobileNetV2, 2 classes, FedAvg,
3 planned rounds, 2 hospitals (`HOSP_999999` with the NIH PNGs, 22,046 training images; `HOSP_888888` with a Roboflow
JPG re-export of NIH cells, 10,521 training images). Rounds 2 and 3 were opened automatically after aggregation and
the job finished as `COMPLETED`. Evaluated on `evaluation_datasets/Malaria/v1` (2,333 images, section 15), one run each:

| Global model | Accuracy | Macro F1 | Parasitized recall | Uninfected recall |
|---|---|---|---|---|
| v1 (round 1) | 49.9% | 0.33 | 0.0% | 100.0% |
| v2 (round 2) | 93.8% | 0.94 | 87.8% | 99.8% |
| v3 (round 3, promoted to `MAIN`) | 96.5% | 0.97 | 94.6% | 98.5% |

v1 predicted every image as Uninfected. On 60 sampled evaluation images the same evaluation path scored
`HOSP_999999`'s own round-1 model and v3 60/60 and `HOSP_888888`'s round-1 model 42/60 (it missed 18 of 30
Parasitized), so this is the averaged model, not the evaluator.

Cause (found 2026-10-16 with the Flower bench, section 24.3): the averaged weights were fine, but the averaged
BatchNorm running statistics did not describe them. Re-measuring v1's BatchNorm statistics on each hospital's own
round-1 training images (50 batches each, forward passes only, averaged by sample count) raised v1 from 49.9% to
95.8% on the same evaluation set, without changing a weight. From round 2 the hospitals started from a shared
trained model and stayed closer together, so the mismatch did less damage. The application protocol does not
calibrate yet (section 20). These are single runs on a public research dataset with an image-level hold-out
(section 15); they are not evidence of clinical performance.

## 19. Security and Integrity Guarantees Actually Implemented
M9 strictly implements and verified the following:
- Participant authorization via central routing expected participants.
- Authenticated JWT identity routing.
- Canonical base-model provenance checking.
- Strict architecture/schema/task topological validation.
- SHA-256 parameter integrity verification and duplicate filtering.
- Safely handling/rejecting NaN/Inf in model tensors.
- SQLite named-column persistence hardening (participants, evaluation records and promotion decisions included since 2026-10-14).
- Join requests with admin approval; only approved hospitals can join rounds (section 6).
- Dynamic historical provenance calculation.

## 20. Known Scope / Future Modules
M9 acts as the foundational lifecycle and provenance pipeline. The following capabilities are outside the current M9 implementation and may be addressed by later security/privacy modules:
- Differential Privacy
- Enhanced secure communication / transport protections
- Byzantine / malicious update detection
- Advanced telemetry and monitoring
- Submitting updates between machines in manual-round jobs: the hospital sends the server the folder path of its update, so hospital and server must share a machine (Flower jobs send the update itself over Flower, section 25)
- Module 12's hook (`federation/security_hooks.py`) currently accepts every update
- More advanced privacy-preserving aggregation mechanisms such as FHE or SMPC, if adopted by the project
- Hospitals cannot yet download the promoted `MAIN` model for local inference (Module 15/16 integration)

Known limitations inside Module 9 (2026-10-16):
- Manual-round jobs average BatchNorm statistics without calibrating them, so a global model can be much
  worse than its weights (v1 of `JOB_MAL_MNV2_CLS2_1`: 49.9%, 95.8% after calibration; section 18). The Flower bench
  (section 24.3) and Flower jobs in the application (section 25) calibrate: hospitals measure statistics for the new
  global model on their own training images and send only the statistics.
- The malaria evaluation set is held out per image, not per patient (section 15). A patient-level hold-out needs
  patients removed from both hospitals' data before training, then a new run.
- In manual-round jobs an update's `training_configuration` stores the hospital's local `dataset_dir` and
  `output_dir` paths (they name the hospital's working folders and the operating-system user). They are not needed
  for aggregation. Flower jobs drop path-like values before the update is stored.
- The `EVALUATION_IN_PROGRESS` badge exists only on the admin screen that started the evaluation; reloading the page
  during an evaluation shows `EVALUATION_PENDING` until it finishes.
- `storage.py` reads jobs, rounds and updates by column position after `SELECT *`. This holds while columns are only
  appended; participants, evaluations and promotion decisions use named columns.

## 21. Key Files
The authoritative logic resides in:
- `federation/models.py`
- `federation/storage.py`
- `federation/validation.py`
- `federation/rounds.py`
- `federation/aggregation.py`
- `federation/api_adapter.py`
- `federation/evaluation.py`
- `federation/cli.py`
- `hospital_client/local_api/server.py`
- `hospital_client/federation/eligibility.py`
- `backend/src/routes/federationRoutes.js`
- `backend/src/controllers/federationController.js`
- `frontend/src/views/AdminFederationModelsView.jsx`
- `federation/base_models.py` (round base model, added 2026-10-09)
- `frontend/src/views/FederatedTrainingView.jsx`, `frontend/src/components/federation/AggregateRoundDialog.jsx`
- `federation/round_plan.py` (round plan, early stopping), `federation/flower_sim.py` (Flower simulation bench), added 2026-10-14
- `backend/src/services/notificationService.js`, `backend/src/routes/notificationRoutes.js`, `backend/src/models/Notification.js`

## 22. Round Notifications (Module 18 integration)

The backend (`backend/src/services/notificationService.js`) stores one notification per active hospital in MongoDB (`Notification` model) and pushes it live to the hospital's Socket.io room `hospital_<hospital_id>`.

| Event | Who is notified | Type |
|---|---|---|
| Admin opens a round, or the round plan opens the next one after aggregation | every active hospital | `ROUND_OPEN` |
| Admin aggregates a round successfully | every active hospital that did **not** join that round | `ROUND_CLOSED` (do not train for this round; you will be notified when the next one opens) |

- Hospitals read them with `GET /api/v1/notifications` and mark them read with `POST /api/v1/notifications/read` (their own hospital only).
- A failed aggregation sends nothing.
- A notification failure never undoes the federation action.

## 23. Final M9 Status
**PASS** — The currently implemented Module 9 federation architecture has been verified against its defined architectural requirements, including hospital participation and authorization, canonical base-model provenance, update validation, weighted aggregation, Global Model creation, persistence, dynamic participant provenance, and Admin-side Global Model retrieval/evaluation integration.

Against the implementation plan (2026-10-14): Flower server/client workflow and multi-client simulation
(`federation/flower_sim.py`), FedAvg, global-model exchange, sample-weighted aggregation, repeated rounds with a
planned total or stopping rule, and FedProx are implemented and tested. The simulation bench has run on the real
malaria data (section 24.4), and a real 3-round application job was run and evaluated (section 18). Since
2026-10-16 the application can also run a job through Flower (section 25), with BatchNorm calibration; this is
verified by automated tests, including an end-to-end run with two hospital services, and by a real 2-round Flower
job on the malaria data in the application (section 25.8). Open inside Module 9: BatchNorm calibration for
manual-round jobs (section 20).
The FedAvg vs FedProx experiments belong to Module 20.

M9 is the completed federation lifecycle and aggregation foundation for the current project scope. Privacy-enhancing mechanisms, enhanced secure-transport protections, Byzantine/malicious-update detection, and advanced telemetry remain outside the current M9 implementation and are reserved for subsequent modules.

## 24. Round Plan, FedProx and the Flower Simulation Bench (added 2026-10-14)

These close the gaps between Module 9 and the implementation plan (Flower workflow, simulation
with several hospital clients, repeated rounds with a stopping criterion, FedProx).

### 24.1 Round plan (application protocol)
Stored in the job's existing JSON fields, validated when the job is created (`federation/round_plan.py`):

| Setting | Where | Meaning |
|---|---|---|
| `strategy` | `aggregation_configuration` | `fedavg` (default) or `fedprox` |
| `total_rounds` | `aggregation_configuration` | Planned number of rounds. Absent = open-ended (previous behaviour). |
| `auto_continue` | `aggregation_configuration` | Open the next round automatically after a successful aggregation. Defaults to on when `total_rounds` is set. |
| `round_duration_hours` | `aggregation_configuration` | Deadline of an automatically opened round. Required with `auto_continue`. |
| `early_stopping` | `aggregation_configuration` | Optional `{metric: accuracy\|f1, min_delta, patience}`: stop when the evaluated global model has not improved by `min_delta` for `patience` rounds in a row. |
| `proximal_mu` | `training_requirements` | FedProx weight (> 0, required for `fedprox`). |

- After aggregation the adapter applies the plan: with early stopping it first evaluates the new
  global model (only evaluated rounds count); when the plan is finished the job becomes `COMPLETED`;
  otherwise, with `auto_continue`, round N+1 opens from the new global model and the backend sends
  `ROUND_OPEN` to every active hospital. The `aggregate` response carries `plan`, `next_round` and `warnings`.
- `create_round` refuses rounds beyond `total_rounds` or after the job finished. Opening the first
  round moves the job from `CREATED` to `ACTIVE`.
- `list_jobs` / `get_job_details` include `plan` (rounds completed, next round number, scores, finished, reason).
- Admin UI: the job form sets total rounds, round duration and early stopping. It does not offer the strategy:
  jobs created there use FedAvg. FedProx is a research setting (choosing it needs knowledge of how different the
  hospitals' data are), available through the API (`aggregation_configuration.strategy`) and the Flower bench.
- Hospitals still train when they choose; "automatic" means the server side (open → collect →
  aggregate → open next) needs no admin action except the aggregate step, which stays an admin decision.

### 24.2 FedProx
- Module 7 `TrainingConfig.proximal_mu` adds `(mu/2)·||w − w_start||²` over the trainable weights
  to the local training objective, where `w_start` is the round's base model. Reported loss and
  metrics stay the plain classification loss. `0` (default) = unchanged FedAvg training.
- The hospital local service passes the job's `proximal_mu` to Module 8/7 (`--proximal-mu`).
- `UpdateValidator` rejects an update of a FedProx job whose training did not use the job's `proximal_mu`.
- Server aggregation is the same sample-weighted average; the global model's `aggregation_metadata`
  records `strategy` and `proximal_mu`.

### 24.3 Flower simulation bench (`federation/flower_sim.py`)
A real Flower server (`flwr.server.start_server`, Flower's own `FedAvg`/`FedProx` strategies) and N
client processes (`flwr.client.start_client`) exchanging parameters over gRPC on 127.0.0.1. No Ray.

```
python -m federation.flower_sim --dataset <Module 5 output dir> --output <results dir> \
    --clients 3 --rounds 5 --partition label_skew --alpha 0.5 \
    --strategy fedprox --mu 0.01 --architecture resnet18 --epochs 1 --batch-size 32 --device cuda --seed 42
```
- Partition: `iid` (seeded shuffle) or `label_skew` (Dirichlet(alpha) share of every class per client).
  Every train/validation record goes to exactly one client; client manifests reference the original files.
- Clients train with the unchanged Module 7 Trainer from the round's global model (checksum-verified
  base-model path), one at a time (shared lock) so a 4 GB GPU is enough.
- The server evaluates the global model on the Module 5 test split after every round (round 0 = start model).
- BatchNorm calibration (added 2026-10-16, bench version 1.1): after every aggregation each client runs the
  averaged model forward over up to `--bn-calibration-batches` (default 50) batches of its own training images to
  re-measure BatchNorm running statistics; the server averages them by sample count. Weights are not changed and
  no images leave the client. Each federated round is then two Flower rounds (train, calibrate), which doubles the
  parameters exchanged. `--bn-calibration-batches 0` gives plain FedAvg/FedProx averaging.

  Why: averaging also averages each client's BatchNorm statistics, which describe that client's weights, not the
  averaged ones. In the first malaria run (MobileNetV2 from scratch, 3 IID clients, 1 local epoch) every client
  model reached 95-96% test accuracy on its own, but the averaged model predicted Uninfected for every image (51.2%)
  in rounds 1 and 2. On three such client models, scored on a fixed sample of 640 test images: plain average 48.4%; average with statistics measured from
  each test batch 93.4% (upper bound, not usable since it uses test data); average with statistics calibrated on
  the clients' training images 93.1%. Averaging three copies of one client gave that client's accuracy unchanged,
  so the averaging itself was correct.
- Output: `results.json` (configuration incl. seed, partition class counts, per-round test metrics,
  client metrics, round durations, bytes exchanged) and `rounds.csv`.
- Limitations: records are split individually (patient groups are not kept together); Flower's FedAvg
  averages BatchNorm's integer counters (the application protocol copies them from the first participant);
  fixed number of rounds (no early stopping in the bench); `start_server`/`start_client` are deprecated
  since Flower 1.13 but need no extra services.

### 24.4 Run on the real malaria data (2026-10-16)
```
python -m federation.flower_sim --dataset <malaria Module 5 output, patient-level split> --output <results dir> \
    --clients 3 --rounds 5 --partition iid --strategy fedavg --architecture mobilenet_v2 \
    --epochs 1 --batch-size 32 --device cuda --seed 42        # lr 1e-3 (Adam), 50 calibration batches
```
NIH malaria cells, Module 5 split by patient (filename prefix), Flower 1.30.0, bench 1.1, RTX 3050 Laptop (4 GB).
Each client got about 7,620 training images (Parasitized/Uninfected roughly 52/48) and about 630 validation images.
Test split: 2,799 images, never used for training or calibration. MobileNetV2 from scratch (no pretrained weights).

| Round | Test accuracy | Test F1 | Parasitized recall | Uninfected recall | Seconds |
|---|---|---|---|---|---|
| 0 (untrained) | 48.8% | 0.33 | 100% | 0% | 11 |
| 1 | 94.2% | 0.94 | 94.0% | 94.4% | 309 |
| 2 | 96.8% | 0.97 | 96.3% | 97.2% | 263 |
| 3 | 96.7% | 0.97 | 97.2% | 96.2% | 252 |
| 4 | 97.0% | 0.97 | 96.8% | 97.2% | 310 |
| 5 | 97.2% | 0.97 | 97.2% | 97.2% | 253 |

Round 5 confusion matrix (rows true, columns predicted; Parasitized, Uninfected): [[1327, 38], [40, 1394]].
About 108.5 MB of parameters were exchanged per round (9.04 MB model, 3 clients, train + calibration, both
directions); 23.4 minutes in total. One run, one seed, IID split: it shows the Flower workflow works end to end on
real data. FedAvg vs FedProx and IID vs label skew comparisons belong to Module 20. Results (`results.json`,
`rounds.csv`) are kept outside the repository with the other malaria outputs.

Jobs with manual rounds (sections 3-22) are unchanged: hospitals still submit through the backend, and the
cross-machine submission limitation in section 20 still applies to them. Jobs created with the Flower protocol run
through the application's own Flower server (section 25).

Tests: `federation/tests/test_round_plan.py` (23), `federation/tests/test_participation.py` (18, including the
evaluation progress file on failure and a locked progress file), `federation/tests/test_flower_sim.py` (13, including BatchNorm calibration,
a failing-client run and a real 2-client FedProx Flower run), `hospital_client/training/tests/test_fedprox.py` (8).
All federation tests: 85 passed; backend: 173 passed, including the evaluation-progress route (2026-10-16).

## 25. Flower in the Application (added 2026-10-16)

A federation job can run its rounds through Flower instead of the admin opening and aggregating each round by hand.
The admin chooses "How rounds run" when creating the job (`aggregation_configuration.protocol`: `"flower"`, the
default for new jobs in the UI, or `"manual"`). Flower jobs need `total_rounds` and always continue automatically.
Existing manual-round jobs behave as before.

### 25.1 What runs where
- **Flower server** (`federation/flower_app/`, started by the backend as `python -m federation.flower_app --job-id ...`
  when the admin presses **Start federation**): one process per job. Its strategy does not replace Module 9; it
  drives it. For every federated round it opens the Module 9 round (or reuses one that is still open), waits for
  approved hospitals, sends the round's base model, takes each hospital's update through the same intake validation
  as a manual round (`IntakeManager`, checksum, base-model provenance, deadline), closes the round and aggregates with
  the Module 9 aggregator. It then runs a BatchNorm calibration step and creates the versioned Global Model.
  Each federated round is two Flower rounds: train, then calibrate.
- **Hospital Flower client** (`hospital_client/federation/flower_client.py`): a thread inside the hospital's local ML
  service. The hospital app connects it automatically for approved Flower jobs while the app is open and the
  hospital is signed in. Each train step starts the same Module 8 training job a manual round starts
  (`build_federated_training`), so the Training Monitor shows it as usual. The base model received over Flower is
  saved as the job's base checkpoint after its checksum is checked.
- **Backend** (`backend/src/services/flowerService.js`): starts and stops the server, reads its status file, turns
  its events into round notifications and gives each UI its view of the status.

Raw images stay at the hospital. Over Flower the hospital sends model parameters, the update's metadata (sample
counts, metrics, training configuration without local paths) and, in the calibration step, BatchNorm running
statistics only.

### 25.2 When a round starts and ends
- A round starts when at least the job's minimum number of approved hospitals are connected and verified. If that
  does not happen before the round's deadline, the round fails with `INSUFFICIENT_PARTICIPANTS` and the run ends.
- The round is aggregated as soon as every selected hospital has sent its update.
- At the deadline, whatever has arrived is aggregated (collection closes `min(120 s, 10% of the round)` early so
  updates in flight still reach Module 9 before its deadline). Fewer accepted updates than the minimum fails the round.
- **Aggregate now** (admin) is enabled once the minimum number of hospitals report their update sent. It writes
  `federation_artifacts/flower/<job>/close_round`; hospitals still training see `closing` and cancel.
- After aggregation the hospitals recalibrate BatchNorm statistics for the averaged weights on up to 50 training
  batches (forward passes only); the server averages them by sample count and saves them into the Global Model.

### 25.3 Identity
The client sends its `hospital_id` and the operator's session token through Flower's `get_properties`. The server
checks the token against the backend (`GET /auth/me`, role `hospital_operator`, matching `hospital_id`, answers cached
60 s) and checks that the hospital is approved for the job. Each update is checked again before intake. Tokens are not
written to the status file, API responses or logs.

### 25.4 Routes and status
Backend (`/api/v1/federation`):
- `POST /jobs/:job_id/flower/start`, `POST /jobs/:job_id/flower/stop`, `POST /jobs/:job_id/flower/aggregate-now` (admin).
- `GET /jobs/:job_id/flower/status`: admins get the full status (every hospital, events, updates sent); a hospital
  gets the job status and its own entry only.
- `POST /jobs/:job_id/flower/report` (hospital): the hospital's state and training progress; returns `closing`.
- `GET /jobs` adds a `flower` summary (state, Flower round, phase, address, `closing`, `stale`, `running`) to Flower jobs.

Hospital local ML service: `POST /api/ml/federation/flower/connect`, `POST .../disconnect`, `GET .../status`.

The server writes `federation_artifacts/flower/<job>/status.json` (atomic replace): state, federated round and
phase, each hospital's step (`connected`, `training`, `sent`, `calibrating`, `done`, `missed`, `rejected`, `failed`) and the
last 200 events. A status not updated for 30 s while the run is not finished is shown as stale.

### 25.5 What the UIs show
- Admin, Federated Training: **Start federation** on a Flower job and a live panel with the server state, the
  current round and step, each hospital's step and training epoch, recent events, **Aggregate now** (with the reason
  when it is disabled) and **Stop**.
- Hospital, Federation: the dataset the job trains with (chosen when the join request was sent) and a step strip
  (connected, model received, training, sending update, waiting for other hospitals, calibrating, round done).
- Hospital, Training Monitor: the same step strip above the usual monitor, which follows the round's training job.

### 25.6 Limitations
- Uses Flower's `start_server`/`start_client` (deprecated since Flower 1.13; no SuperLink/SuperNode).
- No TLS on the Flower connection (`insecure=True`) until Module 11. The token travels in clear text on that link.
- On one machine with one GPU, hospitals train one after another (the training lock is machine-wide), so a round
  takes as long as all hospitals' training together.
- **Stop** ends the Flower server but leaves the current Module 9 round open; **Start federation** continues it.
  The server process belongs to the backend: restarting the backend ends it (the panel then shows it as stale).
- A hospital trains only while its app is open and signed in.
- Flower hands the server a round's updates only when every selected hospital has finished, so until then the
  admin panel learns that a hospital sent its update from the hospital's own report (`POST .../flower/report`),
  not from the server.
- One real application run so far (section 25.8): one seed, two hospitals on one machine.

### 25.7 Tests
- `federation/tests/test_flower_app_engine.py` (5): path stripping, handoff rebuild and checksum, BatchNorm averaging,
  Global Model replacement.
- `federation/tests/test_flower_app.py` (4): two full rounds with real Flower clients, too few hospitals, Aggregate
  now, identity rejected.
- `hospital_client/federation/tests/test_flower_client.py` (7, including reports continuing after training so the
  backend hears the update was sent) and three Flower route tests in
  `hospital_client/local_api/tests/test_server.py`.
- `hospital_client/local_api/tests/test_flower_integration.py` (1): a real Flower server and two hospital local ML
  services on one machine, real Module 8 training on synthetic images, one round to a Global Model (about 5 minutes).
- `backend/tests/api/federation.test.js`: 8 Flower tests (role checks, start refusals, start/stop, status per role,
  stale status, reports and Aggregate now, the job-list summary, round notifications).

### 25.8 Run on the real malaria data (2026-10-16)
Job `JOB_MAL_RES18_CLS2_V2`: Flower protocol, ResNet-18, 2 classes (Parasitized, Uninfected), FedAvg, 2 planned rounds,
24 h deadline per round, 10 local epochs. Two hospitals on one machine (RTX 3050 Laptop, 4 GB), both approved after
their own eligibility checks: NeoPulse Hospital (`HOSP_999999`, 22,046 training images) and BreachCandy Hospital
(`HOSP_888888`, 10,521 training images). The admin pressed Start federation once; everything after that ran by itself.

| Round | Started | Aggregated | Global model |
|---|---|---|---|
| 1 | 14:19 UTC (both hospitals connected 14:22) | 15:07, 2 updates | v2 |
| 2 | 15:07 | 15:49, 2 updates | v3 |

Each round took about 41-48 minutes because the hospitals trained one after another on the shared GPU. Each round
ended as soon as both updates had arrived, well inside its 24 h deadline.

Evaluation on the held-out malaria set (`evaluation_datasets/Malaria/v1`, 2,333 images, held out per image; section 15):

| Model | Accuracy | F1 | Parasitized recall (missed of 1,168) | Uninfected recall (false alarms of 1,165) |
|---|---|---|---|---|
| v2 (round 1) | 96.10% | 0.961 | 95.9% (48) | 96.3% (43) |
| v3 (round 2) | 96.14% | 0.961 | 96.5% (41) | 95.8% (49) |

Round 2 matched round 1 (one more correct image out of 2,333): the model had levelled off after one round on this
data. One run, one seed; these are experimental results, not clinical performance.

Found during the run and fixed: a hospital stopped reporting its step once its training ended, so the admin panel
kept showing it as training (epoch 10 of 10), "updates sent" stayed at 0 and Aggregate now could not be enabled
(the round itself aggregated correctly). Hospitals now keep reporting until the round moves on, and the panel shows
"Update sent". A hospital queued behind another training on the same GPU now says so instead of "Training".
