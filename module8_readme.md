# Module 8: Resource-Aware Training

## 1. Purpose

Module 8 is the hospital/local-side **resource-aware training decision layer**. It detects the
actual local machine (CPU, RAM, GPU/CUDA/VRAM, storage, network), evaluates what that specific
machine can safely support, and dynamically generates up to three training-configuration
recommendations - **Recommended**, **High-Capacity / More Time**, and **Fast / Lower Resource**.
Selecting one produces a real Module 7 `TrainingConfig`, which Module 7 then executes unchanged.

**Module 8 does not train models, does not implement federated learning, and does not perform
research experiments.** It decides *what to run and how*, given the hardware in front of it;
Module 7 (`hospital_client/training/`) still does all the actual training.

Status: **IMPLEMENTED and TESTED** (94/94 tests passing; see §20 for the full/production suite).

## 2. Core Principle

> Detect the actual machine, evaluate what it can safely support, and generate resource-aware
> training configurations dynamically.

This module was explicitly **not** built around the development machine (an NVIDIA RTX 3050
Laptop GPU, 4GB VRAM, 16GB RAM) - that hardware is only one of several profiles this module was
tested against (see §14). Every recommendation is computed from whatever `ResourceProfile` is
detected at run time. Nothing here assumes:
- more GPU/VRAM automatically means a better/larger model should be picked,
- more hardware automatically means more training epochs,
- a fixed model always fills the Recommended/High-Capacity/Fast role,
- the developer's specific GPU is a target to design around.

## 3. Architecture

```
hospital_client/resource_training/
├── policy.py                  # ResourcePolicy - every threshold/margin/candidate/budget (§19)
├── hardware.py                 # raw CPU/RAM/GPU/CUDA/storage/network detection (psutil + torch.cuda + nvidia-smi)
├── resource_profile.py          # ResourceProfile dataclass + build_resource_profile()
├── dataset_characteristics.py    # sample counts/num_classes from Module 5's manifest
├── model_profiles.py              # REAL measured parameter counts via Module 6 build_model()
├── profiler.py                     # REAL dry-run memory + per-batch timing measurement (§7)
├── estimator.py                     # memory estimate + numeric time-RANGE estimate + throughput history
├── evaluator.py                      # ResourceEvaluator: per-model feasibility, all 8 architectures
├── recommender.py                     # <=3 dynamic recommendations + per-model role labeling
├── resolved_config.py                  # RecommendedConfig -> real Module 7 TrainingConfig
├── monitor.py                            # background-thread CPU/RAM/VRAM/GPU-utilization sampling
├── adaptation.py                          # CUDA-OOM detection + batch-size-reduction retry
├── runner.py                               # orchestrates: re-check -> monitor -> Trainer.run() -> adapt -> stats -> M9
├── statistics.py                            # ResourceStatistics (hardware+config+resource+estimation)
├── cli.py, __main__.py                       # `python -m hospital_client.resource_training ...`
└── tests/                                     # 94 tests (simulated hardware tiers + real integration)
```

**Full pipeline** (this is the exact sequence `cli.py`'s `train` command and `runner.py` follow):

```
Start training
      |
      v
M8 resource evaluation        build_resource_profile()  ->  ResourceProfile (this machine, now)
      |
      v
dataset evaluation            inspect_dataset()  ->  DatasetCharacteristics (from Module 5's manifest)
      |
      v
model selection                ResourceEvaluator (real measured param counts + real dry-run
      |                         memory/timing, §6-§7)  ->  ModelAssessment x8  ->
      |                         generate_recommendations()  ->  <=3 RecommendedConfig
      v
training configuration          RecommendedConfig.to_training_config()  ->
      |                          hospital_client.training.config.TrainingConfig (Module 7's own schema)
      v
M7                               Trainer(config)  (Module 7, called as-is, unmodified)
      |
      v
actual training                 Trainer.run()  ->  TrainingResult
      |
      v
resource statistics             ResourceMonitor summary + adaptation log + measured throughput
      |                         ->  ResourceStatistics
      v
M9                               Module 7's own build_federation_handoff(result, output_dir)
                                 ->  FederationHandoff  (when training reached
                                 TRAINING_COMPLETED_AWAITING_FEDERATION)
```

Module 7 is called through its existing public API only (`Trainer(config).run()`, and its own
`build_federation_handoff()`) - there is no subclassing, no monkey-patched hooks, no duplicated
training loop, and no reimplementation of the handoff contract. Module 6 is used only through
`build_model()`, `count_parameters()`, `ARCHITECTURE_CATALOG`, `get_architecture_info()` -
Module 8 never redefines an architecture or a model config schema.

## 4. Hardware Detection (`hardware.py`, `resource_profile.py`)

`build_resource_profile()` detects, on the **actual local machine**:

- **CPU**: processor label, physical/logical core counts, current utilization (`psutil`)
- **RAM**: total/available MB, utilization (`psutil.virtual_memory()`)
- **GPU/CUDA**: `cuda_available`, GPU name, vendor, total/free VRAM (`torch.cuda.mem_get_info()`),
  CUDA compute capability
- **Storage**: total/free space at the configured output path (`psutil.disk_usage()`)
- **Network**: a short-timeout (default 0.5s), non-blocking TCP connect probe recording
  availability and latency - recorded for future Module 9 telemetry only, **never** required for
  or blocking local training

Every detector is isolated: a failure on one axis (e.g. an unreadable storage path) is recorded
as a warning and that section degrades to an explicit "unavailable" placeholder - it never
aborts the whole profile or fabricates a substitute number.

`ResourceProfile` contains hardware/operational information only - **no medical images, patient
records, PHI, raw dataset contents, or model weights** ever appear in it.

## 5. Device Selection (`evaluator.select_device`)

CUDA if `torch.cuda.is_available()`, otherwise CPU - decided **once per machine**, never
per-model. `cuda_available`, the chosen `device`, and (when CPU was chosen instead of CUDA) a
`fallback_reason` are always recorded and never hidden. Module 8 never requires CUDA; CPU-only
machines are fully supported (see §14, very-low profile).

## 6. Resource Evaluation (`evaluator.py`)

`ResourceEvaluator` converts a `ResourceProfile` + `ResourcePolicy` + dataset characteristics
into a `ModelAssessment` for **every one of Module 6's 8 architectures**, always - not just the
ones eventually recommended. Raw measurements (VRAM MB, RAM MB, core counts, measured parameter
counts) drive every decision; a coarse `ResourceTier` label (`very_low_end`/`low_end`/
`medium_end`/`high_end`) is derived only afterwards, purely to size the epoch/time budget - it is
never used to pick a model directly.

For each architecture, the evaluator:
1. Gets the architecture's **real, measured** parameter count (`model_profiles.py` - see §7).
2. Picks the **largest** batch size from `policy.batch_size_candidates` whose estimated memory
   (`estimator.estimate_memory_mb`) fits within a safety-margined budget of the free VRAM (CUDA)
   or available RAM (CPU). If no candidate fits even at the smallest size, the model is
   `feasible=False` ("unsafe") - it is never forced through.
3. Picks a numeric epoch budget (§9) and computes a numeric time estimate (§10).
4. Picks a worker count from `policy.worker_candidates`, reserving CPU cores for the OS/main
   process (never increasing the default worker count on its own).

## 7. Dynamic Model Selection - Real Measured Parameters, Memory, and Timing

Rather than guessing or hard-coding "Model X needs Y GB," Module 8 uses two layers of real
measurement:

1. **`model_profiles.py`** actually constructs each of Module 6's 8 architectures (CPU,
   `pretrained=False`, no gradients) via Module 6's own `build_model()` + `count_parameters()`,
   and caches the real parameter count per `(architecture, num_classes, input_size, color_mode)`.
2. **`profiler.py`** goes further: for the specific candidate batch size the heuristic pre-filter
   says *might* fit, it runs one **real forward + backward + optimizer.step()** on this exact
   machine - mirroring Module 7's own training step exactly (same `autocast`/`GradScaler` call
   pattern as `hospital_client.training.trainer.Trainer._run_epoch`) - using randomly generated
   dummy tensors (never real training data). It measures the **actual** peak memory
   (`torch.cuda.max_memory_allocated()` on CUDA, a process-RSS delta floored at the real
   parameter+gradient footprint on CPU) and the **actual** wall-clock time for that one batch.

If the real measurement exceeds the safety budget even though the cheap heuristic pre-filter let
it through, the evaluator steps down to the next smaller batch-size candidate and measures again
(bounded by `policy.batch_size_candidates`); a real CUDA OOM during the dry run itself is caught
exactly like Module 7 catches one, and treated as "this candidate doesn't fit," never as a crash.

The heuristic in `estimator.estimate_memory_mb()` still exists as a **cheap pre-filter only** (so
obviously-infeasible candidates never need a real dry run) and as the fallback when
`policy.enable_dry_run_measurement=False` or when no candidate survives the pre-filter at all. A
`ModelAssessment.memory_estimate` dict always says explicitly whether its `estimated_total_mb`
was `"measured": true` (real dry run) or a heuristic approximation.

## 8. The Three Recommendations (`recommender.py`)

`generate_recommendations()` evaluates all 8 architectures and, when feasible ones exist, returns
up to three:

| Option | Purpose | Typical Trade-off |
|---|---|---|
| **Recommended** | Balanced resource usage and training capacity for *this* machine | Balanced time and resource consumption |
| **High-Capacity / More Time** | Uses more of the available resources when safely feasible | Longer training and higher resource load |
| **Fast / Lower Resource** | Minimizes training time/resource usage | Lower model/training capacity and potentially different experimental performance |

These are **resource/training-time alternatives, not accuracy rankings.** Module 8 never claims
to know in advance which configuration will produce the best medical-imaging accuracy - that
determination belongs to Module 20 (Research Experiments and Evaluation, §16).

**Selection logic:**
- **Fast** = the feasible architecture with the lowest estimated training time.
- **High-Capacity** = the feasible architecture with the largest measured parameter count that
  still safely fits (i.e. the strongest model the hardware can support, not a fixed model name).
- **Recommended** = the model, from the remaining feasible pool, whose own project-defined
  resource tier is closest to *this machine's own detected capability tier* (§6), tie-broken by
  a capacity-vs-time balance score. This is what makes Recommended genuinely track the hardware:
  a very constrained machine trends toward its lightest feasible tier, a very strong machine
  trends toward a heavier one - it is not a static param-count ranking.

**Honesty about fewer than three.** If only 1 or 2 architectures are genuinely feasible, Module 8
returns that many recommendations plus an explanatory note in `RecommendationSet.notes` - it
**never invents an unsafe third option** to pad the count. If zero architectures are feasible,
it returns an empty list with a note, and every `ModelAssessment.status_label` is set to
`"Unsafe"`.

Every `RecommendedConfig` carries a `reason` and a `tradeoff` string **generated from the actual
computed numbers for this machine and this dataset** (never a canned template) - e.g. "Measured
peak VRAM: approximately 1.63 GB of 4.00 GB available" (real dry-run measurement, §7) or
"Estimated peak RAM: approximately 0.42 GB of 6.20 GB available" (heuristic fallback) - and the
wording is deliberately restricted (verified by tests, see §20) to never claim "best accuracy,"
"guaranteed," "clinically superior," or similar unsupported statements. Allowed language instead:
"Recommended balanced configuration," "may provide lower experimental performance," "actual
performance must be evaluated on the target dataset." Estimated training time is always presented
as a **range** (e.g. "34-43 seconds"), never a single falsely-precise number - see §11.

## 9. EfficientNet-B0: An Important, Never-Forced Reference Model

EfficientNet-B0 is already Module 6's initial primary architecture and is treated as an
important project reference model here too - but it is **never forced** into an unsafe or
inappropriate slot. `ResourceEvaluator.evaluate_all()` always assesses it (alongside the other
7 architectures), and `recommender.py` always assigns it a `status_label`
(`Recommended` / `High-Capacity` / `Fast` / `Suitable` / `High Load` / `Unsafe`) with a dynamic
reason, **regardless of whether it was actually picked** - so it always remains visible in
`RecommendationSet.all_model_assessments["efficientnet_b0"]`.

Its role changes with the detected hardware and dataset, exactly as the project intends:
- On very constrained hardware, it may be feasible but labeled `"High Load"` or `"Suitable"`
  rather than picked, because lighter architectures dominate Fast/Recommended.
- On hardware/dataset combinations where it is the best balance for that machine's tier, it
  becomes `"Recommended"`.
- If no lighter architecture is feasible and it is the strongest one that still fits, it becomes
  `"High-Capacity"`.
- If it is ever genuinely infeasible under the configured safety margins, it is labeled
  `"Unsafe"` and is not selectable - never silently forced through.

Verified across all 4 simulated hardware tiers and multiple hand-constructed selection scenarios
in `tests/test_recommender.py` (including tests that prove it *can* win each of the three roles
when the numbers favor it, and that it is correctly excluded when unsafe).

## 10. Batch Size, Epochs, Precision, Workers

- **Batch size** (`policy.batch_size_candidates`, default `[1,2,4,8,16,32,64]`): the largest
  candidate whose estimated memory fits the safety-margined budget for *this* model on *this*
  machine - never a single fixed value, and never blindly maximized past what fits.
- **Epochs** (`policy.epoch_min`/`epoch_default`/`epoch_max`, `policy.time_budget_minutes_by_tier`):
  starts from the default, then is capped so that `epochs * estimated_seconds_per_epoch` stays
  within the machine's own time budget for its capability tier - **not** "more VRAM -> more
  epochs." A powerful GPU running a very heavy model can still receive a moderate epoch count if
  that model's per-epoch cost is high; a weak machine running a light model can still receive a
  meaningful epoch count if it trains quickly. Module 7's own early stopping (`--early-stopping`)
  remains available and unaffected once a `TrainingConfig` is resolved.
- **Precision**: `fp16` is only ever offered when the selected device is CUDA
  (`policy.prefer_fp16_on_cuda`); CPU configurations always resolve to `fp32` - consistent with
  Module 7's own `precision` validation, which rejects `fp16` on CPU.
- **Workers** (`policy.worker_candidates`, default `[0,2,4]`): the largest candidate that leaves
  `policy.reserved_cpu_cores` free for the OS/main process; the default remains resource-friendly.

## 11. Numeric Training-Time Estimation - Always a Range (`estimator.py`)

A single point-value time estimate is always false precision, so Module 8 **never returns just
one number**. Every recommendation carries `estimated_training_time_seconds`/`_minutes` (a
central value, kept only for epoch-budget arithmetic and backward compatibility) **and**
`..._seconds_min`/`_max` and `..._minutes_min`/`_max`, plus a human-readable
`estimated_training_time_display` that is itself a range (e.g. `"1.5-2.2 minutes"`, or a single
value like `"3 seconds"` only when the bounds round to the same displayed value) - never a bare
"low/medium/high" label. `estimation_method` and `estimation_confidence` say exactly how the
range was produced, from least to most trustworthy:

- **`baseline_estimate`** (`confidence="low"`, widest range - `policy.baseline_range_low/high`,
  default 0.6x-1.8x): used only when a real dry-run measurement could not be obtained at all
  (e.g. `policy.enable_dry_run_measurement=False`). A heuristic derived from the architecture's
  *measured* parameter count (relative to a configured reference), image resolution, dataset
  sample count, batch size, epochs, device type, and precision.
- **`dry_run_estimate`** (`confidence="medium"`, `policy.dry_run_range_low/high`, default
  0.85x-1.35x): the normal case. A **real** forward+backward+optimizer-step was just measured on
  THIS machine for this exact architecture/batch/device/precision (`profiler.py`, §7) and
  extrapolated over the full dataset and epoch count. This is a genuine local measurement, not a
  cross-machine heuristic - it is still a range because a single dry-run batch does not capture
  full-epoch effects (DataLoader warmup, disk I/O variance, thermal throttling, etc.).
- **`measured_hardware_estimate`** (`confidence="medium"` with 1 sample, `"high"` once
  `>= policy.high_confidence_history_samples`): once `HistoricalThroughputStore` has one or more
  **real** measured `samples_per_second` values for the same `(architecture, device, precision)`
  key - recorded only by `runner.py` after an actual completed Module 7 run finishes - future
  estimates use the real average, and the range is built from the **actual observed min/max**
  of historical samples once 2+ exist (a single sample gets a modest configured band instead). No
  historical entry is ever seeded or fabricated; the store starts empty and only grows from
  genuine completed runs.

**Fixed overhead.** Every tier also adds `policy.fixed_overhead_seconds` (default 20s) once per
run, to both bounds. A per-batch/per-epoch formula alone systematically under-counts model
construction, DataLoader startup, and checkpoint I/O - confirmed by real verification on this
machine: a 25-sample/2-epoch run had ~31s of actual duration against a ~2s pure per-batch
extrapolation, i.e. ~29s of overhead that a naive formula would have missed entirely. This
constant is deliberately conservative and not tuned per-dataset-size; it shrinks in relative
importance automatically as `measured_hardware_estimate` data accumulates, since that tier
reflects real end-to-end run timing rather than a per-sample formula.

Module 8 never claims that training time can be predicted exactly before training; every value
is explicitly a range, clearly distinguished from `ResourceStatistics.
actual_training_duration_seconds` (§13) once a run actually completes.

## 12. Runtime Monitoring (`monitor.py`)

`ResourceMonitor` samples CPU utilization, system RAM used, (on CUDA) reserved VRAM, and (on
CUDA) real GPU compute utilization percentage on a background thread at a configurable interval
(`policy.monitor_interval_seconds`, default 1s) **while Module 7's `Trainer.run()` executes
normally in the foreground** - Module 7's training loop is not modified, wrapped, or subclassed
in any way.

GPU utilization is obtained via the `nvidia-smi` CLI (`hardware.detect_gpu_utilization()`,
`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`), which ships with every
NVIDIA driver install - **no new Python dependency** (e.g. `pynvml`) was added. This is a real
measurement, confirmed working on this development machine's driver. On a machine without an
NVIDIA driver/CLI present, the probe fails fast and the monitor honestly reports
`gpu_utilization_available=False` / `peak_gpu_utilization_percent=None` rather than fabricating a
number - the failure is cached after the first attempt so it is not retried every sampling tick.

## 13. Runtime Adaptation and Resource Statistics (`adaptation.py`, `runner.py`, `statistics.py`)

`run_resource_aware_training()`:
1. **Pre-training re-check**: re-detects the machine and confirms the configured device is still
   available. If the config requires CUDA and CUDA has since become unavailable, it raises
   `ResourceCheckFailed` immediately - **it never silently substitutes CPU** for a configuration
   that explicitly asked for CUDA.
2. Starts `ResourceMonitor`, then calls `Trainer(config).run()` exactly as Module 7 already
   works.
3. If the returned `TrainingResult` is a CUDA-OOM failure (`adaptation.detect_cuda_oom` - status
   `TRAINING_FAILED` with an "out of memory" message, matching Module 7's own OOM reporting) and
   fewer than `policy.max_adaptation_attempts` retries have happened, the batch size is halved
   (`adaptation.adapt_config_for_oom`, floored at 1, architecture/device never changed) and the
   run is retried. Every attempt is recorded as an `AdaptationEvent` (timestamp, reason, old/new
   batch size, outcome). Retries are strictly bounded - never infinite - and if batch size cannot
   be reduced further, adaptation stops and the failure is reported as-is.
4. Stops the monitor, computes real measured throughput from the actual completed epochs/samples/
   duration (when available), records it to `HistoricalThroughputStore`, and assembles a
   `ResourceStatistics` object: hardware snapshot, the training configuration actually used,
   monitor summary (peak VRAM/RAM/GPU-utilization, CPU stats), the adaptation log, actual vs.
   estimated duration and its error, and the CUDA-OOM event count. Contains hardware/config/
   resource numbers only - never medical images, patient records, or raw dataset contents.
5. **-> M9**: if `result.ready_for_federation` (status `TRAINING_COMPLETED_AWAITING_FEDERATION`),
   calls Module 7's own, unmodified `build_federation_handoff(result, output_dir)` and returns the
   resulting `FederationHandoff` alongside the result and statistics
   (`(TrainingResult, ResourceStatistics, Optional[FederationHandoff])`). Module 8 implements no
   part of Module 9 itself here - it only carries the pipeline through to the handoff Module 7
   already knows how to produce; `round_id`/`client_id` are left `None` since Module 8 has no
   federation/identity context to supply them from (never fabricated).

**Important, honest limitation**: adaptation here operates at **run/retry granularity**, not
intra-epoch - Module 7's training loop was not modified to support mid-epoch intervention (out
of scope: "do not redesign Module 7"). A CUDA OOM is only detected and adapted-to after Module 7
has already stopped that attempt and reported `TRAINING_FAILED`; Module 8 then retries the whole
run with a smaller batch size rather than resizing batches mid-training.

## 14. Low-Resource Compatibility and Tested Hardware Profiles

Verified via `tests/resource_training/tests/` against **simulated** hardware profiles (clearly
labeled as simulated in `conftest.py` - never presented as physical measurements):

- **Very low**: CPU-only, 4GB RAM
- **Low**: 4GB VRAM GPU, 16GB RAM
- **Medium**: 8GB VRAM GPU, 32GB RAM
- **High**: 16GB+ VRAM GPU, 64GB RAM

...and against the **actual physical development machine** (genuine NVIDIA GeForce RTX 3050
Laptop GPU, 4GB VRAM, 16GB system RAM, confirmed via `torch.cuda.get_device_name(0)` /
`torch.cuda.mem_get_info()`) through a real end-to-end integration test (§15). CPU-only
operation is fully supported and requires no CUDA at any point.

## 15. Module 7 Integration

```python
from hospital_client.resource_training.dataset_characteristics import inspect_dataset
from hospital_client.resource_training.resource_profile import build_resource_profile
from hospital_client.resource_training.recommender import generate_recommendations
from hospital_client.resource_training.resolved_config import to_training_config
from hospital_client.resource_training.runner import run_resource_aware_training

profile = build_resource_profile(storage_path=output_dir)          # M8 resource evaluation
dataset = inspect_dataset(dataset_dir)                              # dataset evaluation
recommendations = generate_recommendations(profile, policy, dataset)  # model selection

chosen = recommendations.recommendations[0]  # or filter by .recommendation_type
training_config = to_training_config(chosen, model_id, dataset_dir, output_dir, dataset.num_classes)  # training configuration

result, stats, handoff = run_resource_aware_training(training_config, policy)  # M7 -> actual training -> resource statistics -> M9
```

`to_training_config()` produces a real `hospital_client.training.config.TrainingConfig` -
Module 8 does not invent a parallel "ResolvedTrainingConfig" schema; once a recommendation is
selected, it *is* the exact object Module 7 already knows how to validate and execute
(`validate_training_config()` passes on every recommendation Module 8 ever produces - verified
in `tests/test_resolved_config.py`).

### CLI

```
python -m hospital_client.resource_training detect --output <dir>
python -m hospital_client.resource_training recommend --dataset <module5_output_dir> --output <dir> [--policy <policy.json>]
python -m hospital_client.resource_training train --dataset <dir> --output <dir> --model-id <id> --choice recommended|high_capacity|fast [--policy <policy.json>]
```

## 16. Module 9 Boundary

Module 8 does **not** implement Flower, a federated client/server, FedAvg/FedProx, aggregation,
or any network transmission. It only prepares a resource-aware `TrainingConfig`, and after a run,
a `ResourceStatistics` record and (when applicable) a `FederationHandoff` obtained by calling
Module 7's own existing `build_federation_handoff()` unchanged (§13, step 5). Module 8 does not
transmit the handoff anywhere, does not implement rounds/aggregation/strategy, and does not know
about Flower - producing the handoff object is as far as this pipeline goes. What a future
Module 9 does with the `TrainingConfig`/`ResourceStatistics`/`FederationHandoff` is entirely
Module 9's design decision.

## 17. Module 20 Boundary

Module 8 records resource **facts** (hardware, configuration, timing, throughput) - it performs
no controlled experiments, no model comparison, no resource benchmarking analysis, and no
research reporting. It never claims clinical validity or that any configuration is medically
superior. That analysis belongs to Module 20 (Research Experiments and Evaluation), which is
expected to consume `TrainingResult`/`ResourceStatistics` as raw input.

## 18. Privacy

All hardware detection and resource statistics are local-only. `ResourceProfile` and
`ResourceStatistics` never contain medical images, patient records, PHI, raw dataset contents, or
model weights - only hardware/operational and training-configuration numbers. The network check
(§4) is a best-effort connectivity probe recorded for future telemetry; it never transmits
anything and is never required for local training to proceed.

## 19. Configurable Policy (`policy.py`)

Every threshold this module uses is a field on `ResourcePolicy` (safety margins, batch-size
candidates, epoch bounds and time budgets, hardware-tier thresholds, activation/optimizer memory
coefficients, baseline throughput references, confidence thresholds, recommendation-scoring
weight, monitoring interval, adaptation limits, network-check parameters, the
`enable_dry_run_measurement` toggle, GPU-utilization probe timeout, and all of the time-estimate
range multipliers and the fixed per-run overhead constant - §11) - nothing is scattered as a
magic number elsewhere in the package. `ResourcePolicy.to_dict()`/`from_dict()` support
saving/loading a custom policy as JSON (`--policy <file>` on the CLI).

## 20. Testing (actually executed)

```
python -m pytest hospital_client/resource_training/tests/ -q          -> 94 passed
python -m pytest hospital_client/dataset/tests/ -q                    -> 21 passed  (unchanged)
python -m pytest hospital_client/preprocessing/tests/ -q              -> 84 passed  (unchanged)
python -m pytest hospital_client/model_management/tests/ -q           -> 144 passed (unchanged)
python -m pytest hospital_client/training/tests/ -q                   -> 165 passed (unchanged)
python -m pytest hospital_client/ -q                                   -> 508 passed, 0 failed

python -m pytest hospital_client/resource_training/tests/ --cov=hospital_client.resource_training --cov-report=term-missing -q
  -> production code (excluding tests/): 917 stmts, 30 miss, 97%
```

Covers everything the original 84-test suite covered (hardware detection, device selection,
resource evaluation across all 4 simulated tiers, the recommendation algorithm's edge cases,
EfficientNet-B0 in every role, no banned accuracy phrases, monitoring, CUDA-OOM adaptation, the
CLI, and the real M4->M5->M6->M8->M7 integration test), **plus**, added in this enhancement pass:
real dry-run memory/timing measurement (`test_profiler.py` - real CPU measurement always, real
CUDA measurement and a real CUDA-OOM propagation check via skipif); the decision-determinism test
explicitly isolated from real measurement timing jitter; time-estimate ranges and the three
estimation-method tiers (`baseline_estimate`/`dry_run_estimate`/`measured_hardware_estimate`);
real GPU utilization via `nvidia-smi` (a real measurement test skipif-gated on CUDA availability,
plus a monkeypatched graceful-degradation test simulating a machine without the NVIDIA CLI); and
the `-> M9` `FederationHandoff` step (verified for real in the M4->M5->M6->M8->M7 integration
tests, and stubbed out in the fast fake-`Trainer` adaptation-retry unit tests since those construct
`TrainingResult`s with no real Module 6 checkpoint on disk to load).

## 21. Limitations (honest, as of this implementation)

The previous version of this document listed the memory estimate, the time estimate, and GPU
utilization monitoring as approximation-only limitations. All three have since been upgraded to
real local measurements (§7, §11, §12). What remains genuinely unresolved:

- **Real dry-run measurement is still an approximation of a full training run**, not a
  substitute for one. A single forward+backward+optimizer-step captures memory and per-batch
  timing accurately, but not effects that only appear over a full epoch (VRAM fragmentation
  across the dataset, DataLoader worker warm-up, disk I/O variance, thermal throttling on
  sustained load). This is exactly why the time estimate is always a *range*, and why
  `measured_hardware_estimate` (real full-run history) is preferred over `dry_run_estimate` once
  it exists.
- **The fixed per-run overhead constant (`policy.fixed_overhead_seconds`, default 20s) is a
  single configured number, not measured per-dataset-size.** It was calibrated against one real
  observation on this development machine (a tiny 25-sample run) and is deliberately
  conservative; it is not dynamically re-calibrated per run.
- **Runtime adaptation operates at run/retry granularity, not intra-epoch**, because Module 7's
  training loop was intentionally not modified (out of scope: "do not redesign Module 7"). A
  CUDA OOM is only detected and adapted to after Module 7 has already stopped that attempt and
  reported `TRAINING_FAILED`; Module 8 then retries the whole run with a smaller batch size
  rather than resizing batches mid-training. The real dry-run memory check (§7) reduces how often
  this path is even needed by catching most infeasible batch sizes *before* a full run starts,
  but it cannot eliminate the possibility entirely (real training activations can still differ
  slightly from a single dry-run batch under sustained load).
- **GPU utilization monitoring depends on the `nvidia-smi` CLI being present** (bundled with the
  NVIDIA driver on essentially any machine with a supported GPU) - a machine with a non-NVIDIA
  GPU, or an NVIDIA GPU with no driver/CLI installed, will honestly report utilization as
  unavailable rather than fabricate a value; VRAM usage itself is still tracked via `torch.cuda`
  regardless.
- **The network check remains a best-effort, short-timeout TCP probe** by design (spec
  requirement: never require network connectivity for local training) - it is sufficient for
  future telemetry purposes only, not a full connectivity/bandwidth test, and this is intentional
  scope, not an unresolved gap.
- **Real dry-run measurement adds real (bounded) latency to `generate_recommendations()`** - up
  to one real training step per architecture (8 total, occasionally more if a heuristic
  pre-filter pass turns out to be optimistic and a smaller candidate must be measured too). This
  is an accepted, documented trade-off for real numbers instead of pure heuristics; the shared
  test fixture (`policy.enable_dry_run_measurement=False`) is used to keep pure decision-logic
  tests fast, with dedicated tests exercising the real measurement path directly.
