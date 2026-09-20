# Module 7: Local Deep Learning Training

## 1. Purpose

Module 7 is the hospital/local-side **local deep learning training engine**.
It consumes Module 5's prepared dataset, obtains a model from Module 6,
trains it locally with PyTorch, evaluates it, tracks metrics, checkpoints
the best result, and prepares (but does not transmit) a model update
candidate for a future federated learning module.

**Module 7 is the training engine. It is not model management (Module 6),
not resource-aware hardware selection (Module 8), and not federated
learning/aggregation (Module 9).**

Status: **IMPLEMENTED and TESTED** (165/165 tests passing, 98% coverage).
This includes a controlled enhancement pass adding optional mixed
precision, two additional LR schedulers, deterministic DataLoader worker
RNG seeding, and a versioned Module 7 → Module 9 `FederationHandoff`
contract - see §17-20.

## 2. Responsibilities

Local PyTorch training loop, DataLoader construction from Module 5's
manifest contract, forward/backward/optimizer step, learning-rate
scheduling (`reduce_on_plateau`/`step`/`cosine`), optional mixed-precision
training, deterministic DataLoader worker seeding, early stopping,
best-checkpoint selection, resume from an interrupted run,
training/validation/test metrics (accuracy, precision, recall, F1,
per-class metrics, confusion matrix), class-weighted loss, reproducibility
(seed recording), a structured `TrainingResult`, and a versioned
`FederationHandoff` object built from Module 6's `get_parameters()`.

## 3. Non-Responsibilities (explicitly NOT implemented here)

Dataset ingestion/inspection (Module 4); preprocessing, splitting, manifest
generation (Module 5); model architecture definitions, artifact storage,
versioning (Module 6 - reused, not duplicated); hardware detection or
resource-aware model/batch-size/device selection (Module 8); Flower,
FedAvg/FedProx, federated aggregation, or any network transmission (Module
9); differential privacy (Module 10); TLS/secure communication (Module 11);
Byzantine/malicious-update detection (Module 12); platform-wide model
governance (Module 15); inference/prediction execution (Module 16);
desktop/mobile UI, notifications (Modules 2/3/17/18). Verified by `grep` -
**NOT FOUND** anywhere in this package.

## 4. Architecture

```
TrainingConfig (this module)
      |
      v
Module 6 ModelConfig ---> Module 6 registry.build_model() ---> torch.nn.Module
      |
      v
Common Training Engine (trainer.Trainer) <--- Module 5 manifests (dataloader.py)
      |
      v
TrainingResult ---> FederationHandoff (via Module 6 get_parameters())
```

There is exactly **one** training loop implementation (`trainer.Trainer`).
All 8 of Module 6's architectures (`vit_b16`, `efficientnet_b4`,
`efficientnet_b0`, `resnet50`, `resnet18`, `mobilenet_v2`,
`mobilenet_v3_small`, `mobilevit_xxs`) run through it unmodified - selecting
a different architecture only changes which Module 6 factory function
`build_model()` calls. Verified in `tests/test_all_eight_models.py` (16
tests: one train+one parameter-extraction test per architecture).

## 5. Package Structure

```
hospital_client/training/
├── __init__.py, __main__.py, cli.py
├── config.py         # TrainingConfig, validate_training_config()
├── dataset.py          # ManifestImageDataset, class-mapping derivation, transforms
├── dataloader.py         # build_dataloaders() from Module 5 manifests
├── losses.py               # build_loss(): none / manifest / balanced class weighting
├── metrics.py                # compute_classification_metrics() (reuses scikit-learn)
├── checkpoint.py               # Module 7's private resume-state I/O (see §9)
├── result.py                     # TrainingStatus, TrainingResult, format_summary()
├── federation.py                   # Versioned Module 7 -> Module 9 FederationHandoff contract (see §20)
├── trainer.py                      # Trainer - the single common training engine
└── tests/                            # 165 tests
```

## 6. Data Input (Module 5 Contract)

Module 7 reads Module 5's manifest files directly
(`{dataset_dir}/train_manifest.json`, `validation_manifest.json`,
`test_manifest.json`) - it never rediscovers the dataset or re-implements
Module 4/5 logic. Verified against **real** Module 4→Module 5 output in
every test (via `tests/conftest.py::make_preprocessed_dataset`, not
hand-built fake manifests).

Because Module 5 always writes the same manifest shape in both lazy and
materialized modes (only `processed_path` differs - `null` in lazy mode, a
real path in materialized mode), one `ManifestImageDataset` class handles
both: it loads `processed_path` when present, else falls back to
`source_path`. Verified in `tests/test_dataloader.py` for both modes.

**Class mapping**: derived once from the training manifest's observed
labels (sorted, deterministic), never reordered or reinvented per split. A
label appearing in validation/test that never appeared in training raises a
clear error; a class with zero validation/test samples is a recorded
warning, not a failure.

**Augmentation**: only `train_manifest.json`'s
`metadata.augmentations.applied_augmentations` (Module 5's own configured
flags) are applied, and only to the training split. Recognized flags:
`h_flip`, `v_flip`, `rotation` (the examples Module 5's own `ImageConfig`
documents). Validation/test transforms are always deterministic (resize +
tensor conversion only).

**Scope**: only Module 5's image-classification manifest output is
supported. Module 5's tabular (CSV/XLS/XLSX) output is out of scope, since
Module 6's entire model catalog is image-classification only.

## 7. Training Configuration

`TrainingConfig` (mirrors Module 5/6's dataclass + validation convention)
accepts `model_id`, a Module 6 `ModelConfig`, `dataset_dir`, `output_dir`,
`device`, `precision`, `epochs`, `batch_size`, `learning_rate`,
`weight_decay`, `optimizer`, `scheduler` (+ per-scheduler parameters),
early-stopping settings, `class_weighting`, `num_workers`, `random_seed`,
and `resume`. `validate_training_config()` reuses Module 6's
`validate_model_config()` rather than duplicating it.

**Module 8 boundary**: `device`/`batch_size`/`epochs`/`learning_rate` are
plain fields Module 7 *executes*; nothing in this module inspects hardware
to choose them. A future Module 8 is expected to populate the same
`TrainingConfig` fields (or an equivalent `ResolvedTrainingConfig`) before
handing it to `Trainer`.

## 8. Loss and Class Imbalance

`CrossEntropyLoss`, with `class_weighting`:
- `"none"` - unweighted (default).
- `"manifest"` - reuses Module 5's already-computed
  `train_manifest.metadata.balancing.class_weights` (raises a clear error
  if Module 5's `imbalance_strategy` was `"none"`, rather than silently
  falling back).
- `"balanced"` - Module 7 computes inverse-frequency weights from the
  **training** split's observed label counts only.

Class weighting only affects the loss function; validation/test data and
their sample counts are provably unaffected (verified in
`tests/test_losses.py`/`test_trainer.py`).

## 9. Checkpoints - Two Distinct Kinds (Important)

1. **Model artifact (Module 6)**: weights + `ModelMetadata`,
   safetensors-only, strictly validated on every load. Saved via
   `ModelStore.save_checkpoint()` under `<output_dir>/models/<model_id>/vN/`
   whenever validation improves (or, if validation is disabled, once at the
   end of a completed run). This is the portable artifact meant to reach
   Module 9/15/16 later.
2. **Training state (Module 7, private)**: optimizer/scheduler/
   `GradScaler` state, epoch, best-metric bookkeeping, RNG state, saved as
   `<output_dir>/state/<model_id>_state.pt` via `torch.save(...,
   weights_only=True)`. This is Module 7's own scratch state for resuming
   an interrupted *local* run - it is never exchanged with another module
   or hospital, which is why plain `torch.save`/`torch.load` (not
   safetensors) is appropriate here, unlike Module 6's artifact format
   which must defend against untrusted/exchanged files. Loaded with
   `weights_only=True` regardless, as defense in depth.

Resuming validates that the saved state's architecture and class mapping
match the current run before restoring anything - a mismatch raises
immediately rather than silently changing the architecture.

## 10. Training Loop, Metrics, and Quality Features

Standard loop: forward → loss → backward → optimizer step → optional
scheduler step → validation → checkpoint/early-stopping decision.
Metrics (via scikit-learn, already a project dependency - not
reimplemented): loss, accuracy, macro precision/recall/F1, per-class
precision/recall/F1/support, and a full confusion matrix, computed
separately for train/validation/(optional) test.

- **Early stopping**: configurable metric (`val_loss`/`val_accuracy`/
  `val_f1`), mode, and patience; never driven by test data (enforced by
  config validation: `early_stopping=True` requires
  `require_validation=True`).
- **Best checkpoint**: never assumed to be the final epoch - a new Module 6
  version is saved only when the selected validation metric improves.
- **Scheduler**: `none` (default, unchanged) / `reduce_on_plateau` (stepped
  on validation loss, unchanged default behavior) / `step` (`StepLR`,
  stepped every epoch) / `cosine` (`CosineAnnealingLR`, stepped every
  epoch, `T_max` defaults to `epochs` when unset). All four go through the
  same `Trainer`; no architecture-specific scheduler logic. See §18.
- **Resume**: see §9.
- **Reproducibility**: `torch.manual_seed(random_seed)` before model
  construction and training; the seed is recorded in every
  `TrainingResult`. Verified identical initial weights under the same seed
  (`tests/test_trainer.py`). Full bitwise determinism across all
  hardware/software environments is not claimed - only that the same seed
  reproducibly initializes the model and DataLoader shuffling on the same
  machine/software stack, consistent with PyTorch's own documented
  determinism caveats.

Test evaluation, when requested, runs exactly once at the very end and
never influences checkpoint selection, early stopping, or any
hyperparameter choice.

## 11. CPU / CUDA and OOM Handling

`resolve_device()` (Module 6) is reused, not reimplemented: an explicit
`device="cuda"` request raises immediately if CUDA is unavailable - there
is no silent CPU fallback. This development machine has a real GPU, so the
CUDA path (construction, forward pass, and a simulated CUDA out-of-memory
error) was exercised for real, not skipped or faked
(`tests/test_device_and_resilience.py`).

**CUDA OOM policy**: caught specifically (matched on `"out of memory"` in
the error message, and only for `device.type == "cuda"`), `torch.cuda.
empty_cache()` is called, the failure is recorded in `TrainingResult.errors`
with `status=TRAINING_FAILED`, and training stops. Module 7 never silently
switches device or architecture in response - the caller must retry with an
explicit different configuration. A non-OOM `RuntimeError` is never
swallowed - it propagates normally.

**Interruption**: a `KeyboardInterrupt` during the loop is caught, the
already-saved training state remains valid for resume, and the result's
status is `TRAINING_INTERRUPTED` rather than an uncaught traceback.

## 12. Training Result and Federation Handoff

`TrainingResult` carries `status` (one of `TRAINING_FAILED` /
`TRAINING_INTERRUPTED` / `TRAINING_COMPLETED` /
`TRAINING_COMPLETED_AWAITING_FEDERATION`), architecture, checkpoint
reference, best epoch, duration, training/validation/(optional test)
metrics (each including per-class metrics and a confusion matrix), class
mapping, the full training configuration, seed, warnings, and errors.
`ready_for_federation` is `True` only for the awaiting-federation status.

`format_summary()` produces the human-readable local summary (model,
epochs, best epoch, validation accuracy/F1/precision/recall, duration,
checkpoint reference, status) with an explicit non-clinical disclaimer -
verified present in every summary (`tests/test_result.py`).

`build_federation_handoff(result, output_dir)` loads the checkpointed model
via Module 6, extracts `get_parameters()` (ordered `List[np.ndarray]`), and
returns the versioned `FederationHandoff` described in §20. **Module 7
never transmits this object anywhere, never imports Flower, and performs no
aggregation.**

### Recorded task and preprocessing spec

Each saved checkpoint's Module 6 metadata records `task_type` (`image_classification`) and a versioned
`preprocessing_spec` (`training.dataset.build_preprocessing_spec`): color mode, input size, resize
method/interpolation, normalization (none), value range, channel order, dtype, and an `upstream` block copied
from the Module 5 manifest's `image_preprocessing` metadata (lazy/materialized mode, color mode, target size)
when present. Module 16 (Local Inference) uses it to reproduce and verify the training input pipeline.

## 13. Privacy Boundary

All dataset loading and training happen entirely on the local filesystem
path Module 5 already prepared. Module 7 performs no network I/O of any
kind and uploads nothing. The `FederationHandoff` it prepares is an
in-memory object only - actual transmission is Module 9's future
responsibility, which does not exist in this repository. Module 7 alone
does not constitute complete privacy protection for the platform.

## 14. CLI

```
python -m hospital_client.training train \
    --dataset <module5_output_dir> --model <architecture_id> --num-classes <n> \
    --output <dir> [--model-id <id>] [--input-size H W] [--color-mode RGB|L] \
    [--pretrained] [--dropout <f>] [--device cpu|cuda] [--precision fp32|fp16|auto] \
    [--epochs N] [--batch-size N] [--lr F] [--optimizer adam|sgd] \
    [--scheduler none|reduce_on_plateau|step|cosine] \
    [--scheduler-patience N] [--scheduler-factor F]        (reduce_on_plateau) \
    [--scheduler-step-size N] [--scheduler-gamma F]         (step) \
    [--scheduler-t-max N] [--scheduler-eta-min F]           (cosine) \
    [--early-stopping] [--early-stopping-metric ...] [--class-weighting none|manifest|balanced] \
    [--num-workers N] [--seed N] [--no-validation] [--test] [--resume]

python -m hospital_client.training validate-config <same flags as train>
```

`train` writes `<output>/<model_id>_training_result.json` (the full
`TrainingResult`) in addition to printing the human-readable summary.
`train`/`infer`/`federate` subcommands beyond `train`/`validate-config` are
intentionally absent - inference and federation belong to Modules 16/9.

## 15. Testing (actually executed)

```
python -m pytest hospital_client/training/tests/ -q         -> 165 passed
python -m pytest hospital_client/dataset/tests/ -q          -> 21 passed  (unchanged)
python -m pytest hospital_client/preprocessing/tests/ -q    -> 84 passed  (unchanged)
python -m pytest hospital_client/model_management/tests/ -q -> 144 passed (unchanged)
python -m pytest hospital_client/ -q                         -> 414 passed, 0 failed

python -m pytest hospital_client/training/tests/ --cov=hospital_client.training --cov-report=term-missing -q
  -> TOTAL 1774 stmts, 39 miss, 98%
```

Coverage includes everything in the original 88-test suite (see below) plus
the enhancement-pass suites: `test_precision.py` (18 tests), `test_
schedulers.py` (21 tests), `test_worker_seeding.py` (9 tests), and
`test_federation.py` (26 tests), plus 3 additional CLI tests covering
`--precision`, `--scheduler step|cosine`, and `--num-workers`.

Original coverage (unchanged): all 8 architectures trained one epoch through
the common engine with real forward+backward passes (never only
EfficientNet-B0); lazy and materialized manifest loading; empty/missing
manifest and class-count/label-mismatch error paths; best-checkpoint
selection, early stopping, `ReduceLROnPlateau`, and resume (including
architecture/class mapping mismatch rejection); class-weighted loss in all
three modes; reproducible seeding; a genuine (not faked) CUDA run, a
simulated CUDA OOM on real CUDA verifying `TRAINING_FAILED` + memory
cleanup, and a `KeyboardInterrupt` producing `TRAINING_INTERRUPTED` with a
resumable state; and the CLI's `train`/`validate-config` commands including
error exit codes.

**Real (not fabricated) verification performed on this development
machine**, in addition to the pytest suite, via a standalone script run
through the full Module 4→5→6→7 pipeline on a genuine synthetic image
dataset:

- CPU + `precision="fp32"` + `cosine` scheduler + `num_workers=2` - trained
  3 epochs, produced validation and test metrics, and generated a valid
  `FederationHandoff` (`protocol_version=1`, checksum-verified).
- CUDA (`NVIDIA GeForce RTX 3050 Laptop GPU`, confirmed via
  `torch.cuda.get_device_name(0)`) + `precision="fp16"` + `step` scheduler -
  trained 2 epochs using real `autocast`/`GradScaler`, produced validation
  metrics, and generated a valid `FederationHandoff` with `device="cuda"`,
  `precision="fp16"`.
- `FederationHandoff.save()` → `FederationHandoff.load()` round-trip to
  disk, with parameter arrays verified byte-identical (`np.array_equal`)
  after reload.

No CUDA behavior was simulated or assumed - the FP16/CUDA verification ran
on real hardware. CPU-only environments will skip the CUDA-marked tests
(`pytest.mark.skipif(not torch.cuda.is_available(), ...)`) cleanly rather
than fail or fabricate a result.

## 16. Limitations

- Single-GPU only; no multi-GPU or distributed training (out of scope by
  design - see §17).
- Mixed precision, when enabled, has been verified to run correctly
  end-to-end on this machine's GPU; no throughput/memory benchmarking was
  performed, so no speed-up or memory-reduction claim is made either way.
- DataLoader worker RNG seeding (§19) makes shuffling and per-worker
  augmentation reproducible given the same seed and worker count on the
  same machine/software stack; it does not, and does not claim to,
  guarantee bit-for-bit identical results across different hardware, OS,
  PyTorch/CUDA versions, or GPU non-determinism (e.g. cuDNN algorithm
  selection).
- Tests establish functional correctness (construction, forward/backward,
  metrics, checkpoint/resume mechanics) on tiny synthetic data - they do
  not and cannot establish real medical-imaging accuracy; that is Module
  20's (Research Experiments and Evaluation) responsibility - see §21.
- No mechanism yet exists to actually receive a Module 6 "global federated
  model" for continued local training - `Trainer`/`get_model` can already
  load any valid Module 6 checkpoint by `(model_id, version)`, but a
  round-trip convention (global model in → local fine-tuning) is Module 9's
  future design, not Module 7's.
- Notification of training completion/failure to a user interface is not
  implemented here - see §21.
- Checkpoints saved, and Module 5 manifests written, before the `task_type`/
  `preprocessing_spec`/`image_preprocessing` metadata existed carry no spec or
  `upstream` block, so Module 16 cannot verify their preprocessing.

## 17. Mixed Precision (Optional)

`TrainingConfig.precision` accepts `"fp32"` (default), `"fp16"`, or
`"auto"`. Resolution happens once per run, in `resolve_precision(precision,
device)`:

- `"fp32"` always resolves to `"fp32"`, on any device - this is the
  unchanged, always-safe baseline; nothing changes for existing configs
  that don't set `precision` at all.
- `"fp16"` resolves to `"fp16"` only on a CUDA device; requesting `"fp16"`
  on CPU raises `ValueError` immediately (at config validation, and again
  as a runtime backstop in `Trainer.run()`) rather than silently training
  in FP32 or crashing deep inside the loop.
- `"auto"` resolves to `"fp16"` on CUDA and `"fp32"` on CPU - it never
  changes which device is used, only which precision runs on the device
  already selected.

Implementation uses the current (non-deprecated) PyTorch AMP API:
`torch.amp.autocast(device_type=..., dtype=torch.float16, enabled=...)`
around the forward pass, and `torch.amp.GradScaler(device=...,
enabled=...)` around the backward/optimizer step
(`scaler.scale(loss).backward()`, `scaler.step(optimizer)`,
`scaler.update()`). `GradScaler`'s documented `enabled=False` no-op
behavior means FP32 training runs through the exact same code path as
FP16, so enabling mixed precision cannot change FP32 behavior.

`GradScaler` state is included in the private training-state checkpoint
(§9) and restored on resume, so an interrupted FP16 run resumes with the
correct loss-scale rather than restarting scale-finding from scratch.

Enabling `"fp16"` does not change the default batch size, worker count, or
any other resource setting, and `"fp32"` remains the default - mixed
precision is opt-in only.

## 18. Learning-Rate Schedulers

`TrainingConfig.scheduler` accepts `"none"` (default, unchanged - no
scheduler is constructed), `"reduce_on_plateau"` (unchanged existing
behavior - stepped once per validation on `val_loss`), `"step"`
(`torch.optim.lr_scheduler.StepLR`, parameters `scheduler_step_size`/
`scheduler_gamma`, stepped once per epoch), and `"cosine"`
(`torch.optim.lr_scheduler.CosineAnnealingLR`, parameters
`scheduler_t_max` (defaults to `config.epochs` when unset)/
`scheduler_eta_min`, stepped once per epoch).

`_build_scheduler()` is the single construction point for all four options;
`Trainer.run()`'s stepping logic dispatches on `config.scheduler`:
`reduce_on_plateau` is stepped inside the validation block with the
validation loss as its argument (as `ReduceLROnPlateau` requires); `step`/
`cosine` are stepped unconditionally once per epoch with no argument. The
default remains no scheduler - existing configs that never set `scheduler`
are unaffected.

Scheduler state (`scheduler.state_dict()`) is included in the private
training-state checkpoint (§9) and restored on resume, verified for both
`step` and `cosine` (`tests/test_schedulers.py`) - a resumed run continues
its LR trace rather than resetting it.

Invalid per-scheduler parameters (e.g. `scheduler_step_size < 1`,
`scheduler_gamma <= 0`, negative `scheduler_eta_min`) are rejected by
`validate_training_config()` before training starts.

## 19. DataLoader Worker RNG Seeding

Shuffling and any worker-process randomness (e.g. augmentation) are made
reproducible for a given `random_seed` via two changes, both derived
deterministically from `TrainingConfig.random_seed` - never a hard-coded
constant:

- Each `DataLoader` is given `generator=torch.Generator().manual_seed(
  random_seed)`, controlling the shuffling order.
- When `num_workers > 0`, each `DataLoader` is given `worker_init_fn=
  functools.partial(_seed_worker, base_seed=random_seed)`, where
  `_seed_worker` seeds Python's `random`, NumPy, and PyTorch inside that
  worker process using `derive_worker_seed(base_seed, worker_id) =
  (base_seed + worker_id) % 2**32` - a distinct, reproducible seed per
  worker. `_seed_worker` is a module-level function (not a lambda/closure)
  specifically so it remains picklable under Windows' `spawn` multiprocessing
  start method; this was verified to work with `num_workers=2` on this
  Windows development machine, both via the pytest suite and a standalone
  `if __name__ == "__main__":`-guarded script (`spawn` requires a real
  module entry point - a `python -c`/stdin script cannot re-exec workers).

**What this does and does not guarantee**: the same `random_seed` and the
same `num_workers` reproducibly determine shuffling order and worker-local
RNG state on the same machine/OS/PyTorch build. This is *not* a claim of
full bit-for-bit determinism across different hardware, operating systems,
PyTorch/CUDA versions, or in the presence of GPU-level non-determinism
(e.g. non-deterministic cuDNN kernels), consistent with PyTorch's own
documented reproducibility caveats. The default `num_workers` remains `0`
(unchanged, single-process, most memory/resource-friendly) - opting into
multiple workers is unaffected in behavior other than gaining this
seeding.

## 20. Module 7 → Module 9 FederationHandoff Contract

`hospital_client/training/federation.py` defines a versioned,
explicitly-serializable contract for handing a completed local training
result to a future Module 9 (Federated Learning Engine). **This module
implements only the contract/data structure - no Flower integration, no
FedAvg/FedProx, no federated rounds, no network transport, no aggregation,
and no security/privacy layer.** Those remain Module 9/10/11/12's
responsibility.

**Fields** (`FederationHandoff` dataclass):

| Field | Meaning |
|---|---|
| `protocol_version` | Currently `1`. `SUPPORTED_PROTOCOL_VERSIONS = {1}`; an unsupported version fails validation rather than being silently accepted. |
| `model_id`, `model_version`, `architecture` | Identify the Module 6 artifact these parameters came from. |
| `parameters` | Ordered `List[np.ndarray]`, produced by Module 6's existing `get_parameters()` - not a new parallel extraction mechanism. |
| `parameter_count`, `parameter_shapes`, `parameter_dtypes` | Recorded metadata about `parameters`, cross-checked on validation/load. |
| `parameters_checksum` | SHA-256 over the ordered raw bytes of all parameter arrays - detects tampering/corruption. |
| `num_train_samples`, `num_classes`, `class_mapping` | From the actual training run - never fabricated. |
| `training_metrics`, `validation_metrics` | The `TrainingResult`'s own metrics dicts. |
| `training_configuration` | `TrainingConfig.to_dict()` - full reproducibility record. |
| `device`, `precision` | The resolved device/precision this run actually used. |
| `status` | The originating `TrainingResult.status`. |
| `round_id` | `Optional[str]`, default `None`. Only ever set if the caller explicitly supplies it - Module 7 has no notion of federated rounds and never invents one. |
| `client_id` | `Optional[str]`, default `None`. Same policy - a hospital/client identity belongs to a future identity/auth integration, not Module 7. |
| `generated_at` | Timestamp of handoff construction. |

**Serialization**: raw parameter arrays are **never** embedded in JSON.
`save(directory)` writes two files: `handoff_parameters.npz` (all
parameter arrays, via `np.savez`) and `handoff_metadata.json` (everything
else, via `to_dict()`, which explicitly excludes the raw arrays). Verified
(`test_save_does_not_embed_raw_arrays_in_json`) that the metadata JSON
stays under 10 KB even when the parameters total several megabytes.
`load(directory)` reconstructs and re-validates a `FederationHandoff` from
both files.

**Validation** (`validate_federation_handoff()`), called by both
`build_federation_handoff()` and `load()`: protocol version must be
supported; `model_id`/`architecture` non-empty; `model_version` a positive
integer; `parameter_count` must match `len(parameters)` and the lengths of
`parameter_shapes`/`parameter_dtypes`; every parameter's actual shape/dtype
must match its recorded metadata; the checksum is recomputed and compared
- any mismatch raises `ValueError`; `num_classes` must equal
`len(class_mapping)`; `num_train_samples` must be non-negative. **There is
no silent truncation, reordering, zero-filling, or shape-correction of a
malformed handoff anywhere in this path** - a bad handoff always raises,
verified for parameter-count mismatch, shape mismatch, dtype mismatch,
empty `model_id`, invalid `model_version`, class-mapping mismatch, and
negative sample count (`tests/test_federation.py`).

**No coupling to a transport**: nothing in `federation.py` imports HTTP
clients, Socket.IO, Flower, or MongoDB drivers. It is a plain, versioned,
serializable Python object - how a future Module 9 actually receives one
(HTTP call, message queue, shared filesystem, etc.) is entirely Module 9's
design decision.

## 21. Remaining / Future Integration (NOT part of Module 7 - not completed here)

The following are explicitly out of Module 7's scope and are **not**
marked as completed by this module:

- [ ] **Notification integration.** Module 7 exposes a completed/failed/
  interrupted `TrainingResult` and prints a local summary (§12) - it does
  not push notifications, emails, or UI alerts to any user. Delivering
  "training finished"/"training failed" style notifications to a
  researcher, hospital operator, or mobile user belongs to Module 18
  (Communication and Notification), consuming the `TrainingResult`/
  `FederationHandoff` that Module 7 already produces.
- [ ] **Medical accuracy / research experiments.** Module 7 computes
  standard classification metrics (accuracy, precision, recall, F1,
  per-class metrics, confusion matrix) on whatever dataset it is given,
  including the tiny synthetic datasets used in its own test suite. It
  does **not** perform, and this module makes no claim of, formal research
  experiments, model comparisons, benchmarking against clinical baselines,
  or medical-imaging accuracy/superiority reporting. That analysis,
  reporting, and any clinical-relevance framing belong to Module 20
  (Research Experiments and Evaluation), which is expected to consume
  Module 7's `TrainingResult` outputs as raw input rather than duplicate
  this training engine.

## 22. Completion Checklist (this enhancement pass)

- [x] Common training engine (one `Trainer`, all 8 Module 6 architectures)
- [x] Training/evaluation metrics (accuracy, precision, recall, F1, per-class, confusion matrix)
- [x] Checkpoint/resume
- [x] Reproducibility seed handling
- [x] Early stopping
- [x] Best checkpoint selection
- [x] CUDA OOM handling
- [x] Structured `TrainingResult`
- [x] `FederationHandoff` (versioned - see §20)
- [x] Optional mixed precision (§17)
- [x] Configurable LR schedulers: none/reduce_on_plateau/step/cosine (§18)
- [x] Deterministic DataLoader worker RNG seeding (§19)
- [x] Versioned Module 7 → Module 9 handoff contract (§20)
- [ ] Notification integration (Module 18 - see §21)
- [ ] Medical accuracy / research experiments (Module 20 - see §21)
