# Module 10 — Differential Privacy (DP-SGD)

> **Status**: Completed. Implemented and tested (25 automated tests, plus a real two-hospital run
> on the malaria dataset; see [Verification](#verification)). DP is configured per job through the
> federation API or the admin job-creation form. Known limitations are listed below; the items that
> need further work are collected under [Future Work](#future-work).

---

## Overview

Module 10 adds (ε, δ)-differential privacy to federated training via DP-SGD (Differentially
Private Stochastic Gradient Descent). The implementation uses Opacus 1.6.0 with:

- **Poisson sampling** (`make_private(poisson_sampling=True)`)
- **Per-sample gradient clipping** — each gradient is clipped to `‖g‖₂ ≤ C` before noise is added
- **Calibrated Gaussian noise** — `N(0, σ²C²I)` added to each clipped gradient sum
- **BatchMemoryManager** — decouples logical batch size (sets q and the privacy budget) from physical batch size (GPU memory limit)
- **GroupNorm** architectures — BatchNorm cross-sample statistics are incompatible with Opacus per-sample gradient hooks
- **FP32 training** — PyTorch AMP GradScaler does not compose correctly with Opacus per-sample gradient accumulation

---

## Privacy Parameters

| Parameter | Symbol | Meaning |
|---|---|---|
| `dp_noise_multiplier` | σ | Gaussian noise ratio. Higher σ → stronger privacy. Fixed per job. |
| `dp_max_grad_norm` | C | Per-sample gradient clipping bound. Fixed per job. |
| `dp_target_delta` | δ | Failure probability. Must be ≤ 1/N. Fixed per job. |
| `dp_target_epsilon` | ε_cap | Cumulative ε budget cap across all rounds. Fixed per job. |
| `dp_logical_batch_size` | B | Logical batch size. Each hospital derives q = B/N from its own N. |

The server fixes σ, C, δ, and ε_cap per job. Each hospital derives q from its own dataset size N, plans steps, and verifies projected cumulative ε stays within the cap before training starts. The server re-derives ε independently from (σ, q, steps, δ) and rejects updates where the numbers diverge by more than 1e-3.

---

## RDP Accounting and Multi-Round Composition

Opacus uses the Rényi Differential Privacy accountant (Mironov 2017) with Poisson subsampling amplification. For multiple rounds, the entire history is composed by feeding all (σ, q, steps) tuples to a single RDP accountant:

```python
accountant.history = [(σ₁, q₁, steps₁), (σ₂, q₂, steps₂), ...]
ε_cumulative(δ) = accountant.get_epsilon(delta=δ)
```

This is not a sum of per-round ε values. The RDP accountant exploits log-moment representations to compose tightly over the sequence.

### Server-side ledger (`federation_dp_ledger`)

One row per accepted update, per hospital, per round:

| Column | Type | Description |
|---|---|---|
| `entry_id` | TEXT | `DPL-<hex>` |
| `federation_job_id` | TEXT | Job this update belongs to |
| `round_id` | TEXT | Round this update was accepted in |
| `participant_id` | TEXT | Hospital identifier |
| `noise_multiplier` | REAL | σ reported and verified against job config |
| `sample_rate` | REAL | q = B/N as reported by the hospital |
| `steps` | INTEGER | Opacus optimizer steps taken |
| `delta` | REAL | δ for this round |
| `round_epsilon` | REAL | ε for this round alone (informational) |
| `cumulative_epsilon` | REAL | ε composed over all accepted rounds to date |
| `created_at` | TEXT | ISO-8601 timestamp |

`storage.compute_cumulative_dp_epsilon(job_id, participant_id, delta)` recomputes ε from the full history at any time without relying on the stored `cumulative_epsilon` field.

### Hospital-side state (`dp_state.json`)

In federated runs the file is `<ML_WORK_DIR>/<hospital_id>/federation_runs/<job_id>/dp_state.json`
(default `ML_WORK_DIR` is `~/medfl_ml_work`). Standalone training without `--dp-state-path` writes
`dp_state_<model_id>.json` in the output directory instead.

```json
{
  "job_id": "JOB-...",
  "history": [
    {
      "round_id": "JOB-..._R1",
      "noise_multiplier": 1.1,
      "sample_rate": 0.0640,
      "steps": 30,
      "delta": 1e-5,
      "round_epsilon": 2.7216,
      "cumulative_epsilon": 2.7216
    }
  ]
}
```

Both sides compute cumulative ε from the same history, providing end-to-end auditability.

---

## Budget Enforcement Flow

1. Hospital loads `dp_state.json` → existing history.
2. `project_and_check_budget(history, σ, q, planned_steps, δ, ε_cap)` — raises `PrivacyBudgetExceededError` if projected ε > ε_cap. **Training is refused before it starts.**
3. Training runs under Opacus PrivacyEngine (Poisson sampling + BatchMemoryManager).
4. Hospital appends the round's `(σ, q, actual_steps, δ)` to `dp_state.json`.
5. Hospital reports `(σ, q, actual_steps, δ, ε_round, ε_cumulative)` in `training_configuration`.
6. Server's `UpdateValidator` re-derives ε and checks cumulative ε ≤ ε_cap.
7. If accepted: `IntakeManager` writes exactly one row to `federation_dp_ledger`. `flower_app/server.py` does **not** write a duplicate.

---

## What DP Guarantees

**Formal guarantee**: At the end of all federated training rounds, the mechanism that produced
the sequence of global models satisfies (ε, δ)-differential privacy: for any two datasets D and
D′ that differ in exactly one training image belonging to any single hospital, and for any set of
outputs S:

```
Pr[Mechanism(D) ∈ S] ≤ exp(ε) · Pr[Mechanism(D′) ∈ S] + δ
```

This holds **given** that:

1. σ, C, and Poisson sampling rate q were applied faithfully at every training step at every hospital.
2. The full round history is composed using the RDP accountant (not by adding per-round ε values).
3. δ ≤ 1/N for each hospital's training set of size N.

The guarantee covers every model published from the job — not only the final global model, but also the intermediate global model after each round. Because every round's ε is included in the cumulative composition, the privacy cost is charged at training time, not at publication time.

**Computed values** (σ=1.1, C=1.0, δ=1e-5, B=64, 2 epochs/round, Opacus 1.6.0 RDPAccountant):

| Hospital | N | q | Steps/round | ε after round 1 | ε after round 2 (RDP composed) |
|---|---|---|---|---|---|
| Hospital 1 | 1 000 | 0.0640 | 15 | 2.7216 | 3.4489 |
| Hospital 2 | 1 100 | 0.0582 | 17 | 2.6071 | 3.3033 |

Naive per-round addition (not used) would give 5.4432 and 5.2142 respectively.

---

## What DP Does Not Protect Against

1. **Model inversion / gradient inversion on the update itself.** Opacus adds noise to gradients before the update leaves the hospital. A sufficiently powerful server may still attempt to reconstruct training images from the noised gradient update. Module 11 (encryption in transit, planned, not implemented yet) would prevent the update being read in transit; Module 12 (Byzantine / malicious-update detection) addresses adversarial updates from participants, but neither provides reconstruction resistance. There is no planned secure aggregation module.

2. **Membership inference from the model by an observer who sees per-round model snapshots.** The (ε, δ) guarantee is over the entire published sequence; a model published after each round is covered. However, an observer who can query the model and accumulate signal across many rounds can mount practical membership inference attacks even within the DP guarantee — ε=3 does not mean an attack is infeasible.

3. **Reconstruction of the class label distribution.** The class mapping (label names and integer indices) is part of the job definition and is shared in plaintext. It is not protected by DP-SGD.

4. **Side channels outside the training loop.** Dataset size N is reported in the handoff and used to derive q = B/N. Observers can thus infer N from q and B.

5. **Correctness of the hospital's Opacus invocation.** The server verifies that the reported (σ, q, steps, δ) are self-consistent and that σ matches the job requirement. It cannot verify that the hospital actually ran Opacus rather than reporting plausible parameters while training without noise.

6. **Privacy of the validation and test sets.** DP-SGD applies only to the training data loader. Validation accuracy, per-class metrics returned in the handoff, and test scores computed during evaluation are outside the DP budget.

7. **Class counts, image counts, and eligibility summaries in join requests.** The join-request eligibility payload (class counts per label, total image count, validation metrics) is sent in the clear as part of the participation request. This information is not protected by the DP guarantee and operators should treat it as potentially identifying.

8. **Patient-level privacy.** The guarantee is per training image (neighbouring datasets differ in one image). A patient who contributes k images is protected only at the group level, which is substantially weaker than (ε, δ) for k > 1 (roughly kε, with a correspondingly larger δ). In the malaria dataset each patient contributes many cell images, so ε must not be read as a per-patient bound.

---

## Architecture Compatibility

When `norm_type="group"`, `build_model` applies the following per-instance patches (via `types.MethodType` — **class definitions are never modified**):

| Architecture | Issue | Fix applied |
|---|---|---|
| ResNet-18/34 (`BasicBlock`) | `out += identity` (inplace) | Per-instance forward: `out = out + identity` |
| ResNet-50/101 (`Bottleneck`) | `out += identity` (inplace) | Per-instance forward: `out = out + identity` |
| MobileNetV3-Small (`InvertedResidual`) | `x + self.block(x)` — already out-of-place | Per-instance forward kept as-is |
| All architectures | `inplace=True` on activations | Set `mod.inplace = False` on each module instance |
| All architectures | `BatchNorm*` layers | `ModuleValidator.fix()` replaces them with `GroupNorm(8)` |

Non-DP models built in the same process are not affected: `_resnet.BasicBlock.forward` and `_resnet.Bottleneck.forward` remain the original class methods.

---

## norm_type Propagation

- **Job creation** (`federation/jobs.py`): DP jobs must declare `norm_type="group"`.  `ModuleValidator.is_valid()` is called on the built model; job creation is refused if errors remain after `fix()`.
- **Base model** (`federation/base_models.py`): seed model is built with `norm_type="group"` before weights are stored.
- **Aggregation** (`federation/aggregation.py`): writes `norm_type` into the global model's metadata.
- **Evaluation** (`federation/evaluation.py`): reads `norm_type` from stored metadata.
- **Hospital model build** (`hospital_client/local_api/server.py`): passes `norm_type="group"` for DP jobs.
- **Inference / Module 16** (`hospital_client/model_management/config.py`): `ModelConfig.from_dict()` defaults `norm_type` to `"batch"` when the field is absent, so all pre-M10 checkpoints load without changes.

---

## Flower Integration

DP jobs use one Flower exchange per round (train only). The even-numbered calibration exchange used by non-DP jobs is skipped:

```python
# flower_app/server.py  configure_fit
if self.is_dp:
    return self._configure_train(client_manager)          # train only
return (_configure_train(…) if server_round % 2           # non-DP: odd=train
        else self._configure_calibrate())                 #          even=calibrate
```

GroupNorm has no running mean/var, so calibrating batch statistics would be both unnecessary and a privacy violation (sending un-noised activations to the server).

---

## Configuration (API)

DP is configured at job creation time, through the federation API or the admin job-creation form (DP toggle with σ, C, δ, ε cap, logical batch size and epochs). Hospitals do not choose DP settings: they receive them from the server via the round invitation. The federated training screen shows each DP job's settings and the ε spent per hospital (hospital users see only their own hospital's spend), and the top bar shows how many jobs have DP on.

### Example job creation payload

```json
{
  "training_requirements": {
    "dp_enabled": true,
    "dp_noise_multiplier": 1.1,
    "dp_max_grad_norm": 1.0,
    "dp_target_delta": 1e-5,
    "dp_target_epsilon": 10.0,
    "dp_logical_batch_size": 64,
    "norm_type": "group",
    "precision": "fp32"
  }
}
```

### Hospital CLI

```bash
python -m hospital_client.resource_training \
  --dp-enabled \
  --dp-noise-multiplier 1.1 \
  --dp-max-grad-norm 1.0 \
  --dp-target-delta 1e-5 \
  --dp-target-epsilon 10.0 \
  --dp-logical-batch-size 64
```

---

## Known Limitations

1. **No secure aggregation is planned.** Noised updates are sent in plaintext to the server. Module 11 (planned, not implemented yet) would encrypt the transport layer, so updates currently travel unencrypted. Module 12 (planned) would detect adversarial updates. Neither provides cryptographic aggregation hiding individual updates from the server.

2. **Single-machine DP only.** Opacus `DistributedDPOptimizer` (multi-GPU / multi-process) is not supported. Each hospital must train on a single process.

3. **No FP16.** PyTorch AMP GradScaler does not compose correctly with Opacus per-sample gradient hooks. DP jobs must use `precision=fp32`.

4. **DP + FedProx are mutually exclusive.** The proximal term references un-noised global weights, which is incompatible with per-sample privacy accounting. Both `training/config.py` and `federation/round_plan.py` enforce this.

5. **ε_cap is advisory on the hospital side.** The hospital checks the budget before training using `dp_state.json`. If that file is deleted or corrupted, it will not refuse a round that would exceed the cap. The server enforces the cap independently using the database ledger.

6. **Epsilon reporting tolerance is 1e-3.** A gap > 1e-3 between reported and recomputed ε causes rejection. This accommodates floating-point rounding differences across OS and BLAS versions.

7. **Per-image, not per-patient, guarantee.** See item 8 under "What DP Does Not Protect Against".

8. **No same-architecture non-DP baseline.** The DP vs non-DP comparison below uses a non-DP ResNet-18 job, not a non-DP MobileNetV3-Small job, so it is indicative only.

9. **Noise cannot be verified by the server.** DP noise is applied automatically by the hospital-side client, using parameters fixed by the admin at job creation; hospital users cannot change or disable it. The guarantee assumes the hospital runs the unmodified client software. The server checks that the reported DP parameters match the job, but cannot verify from the model weights alone that noise was added, because training runs on the hospital's machine.

---

## Verification

### Automated tests

```bash
python -m pytest hospital_client/training/tests/test_dp_training.py federation/tests/test_dp_federation.py federation/tests/test_dp_job_summary.py -q
```

Result (20 Oct 2026): `25 passed`. Covered: pre-training budget refusal, DP training and accounting,
physical-batch adaptation, multi-round state persistence, FedProx and non-FP32 refusal, multi-round
ledger composition, exactly-once ledger charging over Flower, server rejection of updates with DP
disabled or mismatched σ, C, δ, ε, q or precision, round-plan validation, and the per-job DP summary.

### Real federated run: malaria dataset (20 Oct 2026)

Job `JOB_MAL_MNV3_CLS2_V2`: MobileNetV3-Small (GroupNorm, FP32), 2 classes (Parasitized,
Uninfected), FedAvg over Flower, 2 rounds, 2 hospitals. DP settings: σ = 1.0, C = 1.0,
δ = 10⁻⁵, ε cap = 5, logical batch B = 256, 10 local epochs per round.

| Hospital | Training images N | q = B/N | Steps/round | ε round 1 | ε after round 2 (RDP composed) | Projected after round 3 |
|---|---|---|---|---|---|---|
| HOSP_999999 | 22,046 | 0.01161 | 870 | 2.298 | 3.147 | 3.855 (allowed) |
| HOSP_888888 | 10,521 | 0.02433 | 420 | 3.563 | 4.909 | 6.018 (would be refused) |

δ = 10⁻⁵ is below 1/N for both hospitals. The smaller hospital spends more ε per round because
each image has a higher chance of being sampled (larger q).

Checks performed:

- HOSP_999999's `dp_state.json` and the server's `federation_dp_ledger` contain the same σ, q,
  steps, δ and ε values.
- Every ε in the ledger was recomputed independently with Opacus `RDPAccountant` from (σ, q, steps, δ)
  and matched to full float precision.
- The ledger has exactly one row per hospital per round.

Evaluation of the resulting global models on the admin evaluation set `EVAL-MALARIA`
(2,333 images: 1,168 Parasitized, 1,165 Uninfected):

| Global model | After round | Accuracy | F1 | Recall Parasitized | Recall Uninfected |
|---|---|---|---|---|---|
| GM-MALARIA-MOBILENET_V3_SMALL-V2 | 1 | 0.9404 | 0.9413 | 0.9084 | 0.9725 |
| GM-MALARIA-MOBILENET_V3_SMALL-V3 | 2 | 0.9567 | 0.9570 | 0.9384 | 0.9751 |

These are experimental results on a public demonstration dataset, not a clinical evaluation.
The privacy guarantee is per cell image, not per patient (see item 8 under "What DP Does Not Protect Against").

### Comparison with a non-DP job

Baseline: job `JOB_MAL_RES18_CLS2_V2`, ResNet-18 without DP, run on 16 Oct 2026 with the same two
hospitals (32,567 training images in total), FedAvg over Flower, 2 rounds, 10 local epochs per round,
evaluated on the same `EVAL-MALARIA` set.

| | Non-DP (ResNet-18) | DP (MobileNetV3-Small, σ = 1.0) |
|---|---|---|
| Accuracy after round 1 | 0.9610 (GM-MALARIA-RESNET18-V2) | 0.9404 (GM-MALARIA-MOBILENET_V3_SMALL-V2) |
| Accuracy after round 2 | 0.9614 (GM-MALARIA-RESNET18-V3) | 0.9567 (GM-MALARIA-MOBILENET_V3_SMALL-V3) |
| F1 after round 2 | 0.9614 | 0.9570 |
| Recall Parasitized after round 2 | 0.9649 | 0.9384 |
| Recall Uninfected after round 2 | 0.9579 | 0.9751 |
| Privacy spent after round 2 | none (no DP guarantee) | ε = 3.147 (HOSP_999999), ε = 4.909 (HOSP_888888), δ = 10⁻⁵ |

Observations:

- After round 1 the DP model is 2.1 points behind; after round 2 the gap is 0.5 points.
- The DP model misses more Parasitized cells (recall 0.938 vs 0.965), i.e. more false negatives on
  the infected class, while recognising Uninfected cells slightly better.

This comparison is indicative, not a controlled measurement of the cost of DP. The two jobs differ in
more than DP: architecture (ResNet-18 has roughly 11M parameters, MobileNetV3-Small roughly 1.5M) and
normalisation (non-DP jobs use BatchNorm, DP jobs must use GroupNorm). A controlled measurement needs
a non-DP MobileNetV3-Small job with the same settings (see Future Work).

---

## Future Work

1. **Controlled DP vs non-DP comparison.** Run a non-DP MobileNetV3-Small job with the same hospitals,
   rounds and epochs, so that architecture is not a confounding factor. Normalisation would still
   differ (BatchNorm vs GroupNorm) unless non-DP jobs are allowed to use GroupNorm.
2. **Membership-inference audit.** Measure empirically, with and without DP, how well an attacker can
   tell whether an image was in the training set. This tests the effect of DP on the trained model,
   in addition to the accounting.
3. **Patient-level privacy.** Sample and clip per patient instead of per image, so that ε bounds what
   is revealed about a whole patient. Opacus does not provide this directly.
4. **Verifying that noise was added.** Trusted hardware with remote attestation (for example Intel
   TDX, AMD SEV, or GPU confidential computing) or cryptographic proofs of training would let the
   server verify the hospital ran the unmodified DP code. Neither is practical on the current hardware.
5. **Encryption in transit.** Module 11 (TLS) is required so that noised updates are not readable on
   the network.
