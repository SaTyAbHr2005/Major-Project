# Module 12 — Byzantine and Malicious Model-Update Detection & Robust Aggregation

> **Status**: Completed. Implemented and verified (19 automated security tests, plus round plan, storage,
> and end-to-end Flower integration tests; see [Verification](#verification)). Robust aggregation and
> security thresholds are configurable per job through the federation API or admin job-creation form.
> Known limitations and threat boundaries are explicitly documented below; items needing further development
> are cataloged under [Future Work](#future-work).

---

## Overview

Module 12 introduces defense-in-depth against Byzantine, adversarial, and malfunctioning hospital updates
in federated learning. In healthcare federations, an adversarial or faulty client could send poisoned gradients,
exploding weights, or manipulated metadata that degrade the global model or stall convergence.

The defense operates on two complementary tiers:
1. **Intake Security Hook (`M12SecurityHook`)**: Per-update verification executed at update intake before
   acceptance. Evaluates parameter sanity (NaN/Inf), sample-count consistency, L2 parameter delta norms
   over **trainable weights only**, cosine alignment against prior global model trajectories (rounds ≥ 3),
   and plausibility of reported training metrics. Every check that cannot execute explicitly records
   a structured reason in `metrics["checks_skipped"]` and is displayed in the UI (the update is still
   accepted so the federation does not deadlock, meaning skipped checks are fail-open with full auditability,
   not fail-closed).
2. **Robust Aggregation Engine (`Aggregator`)**: Server-side aggregation algorithms — **Coordinate-wise Median**
   and **Trimmed Mean** — alongside standard FedAvg/FedProx. Limits the influence of outliers and flagged
   updates, with strict BatchNorm calibration over accepted participants and automatic fallback when $K < 3$.

---

## Threat Model

Federated learning in healthcare involves independent hospital nodes with local datasets. The threat model
explicitly defines what Module 12 defends against and what is outside its current scope.

### Covered Threats

| Threat | Description | Defense Mechanism |
|---|---|---|
| **Exploding / Corrupted Weights** | Updates containing `NaN` or `±Inf` values from numerical instability or fault injection. | **Parameter Sanity Check**: Immediate `REJECT`. |
| **Extreme Scaling Attack** | Attacker scales parameter delta by a large multiplier (e.g., 5x–50x) to dominate the global model average. | **L2 Delta Norm Bound (Trainable Only)**: `FLAG` or `QUARANTINE`. |
| **Random Noise Injection** | Attacker sends random Gaussian noise vector $\mathcal{N}(0, \sigma^2 I)$ to disrupt convergence. | **L2 Delta Norm Bound**: Exceeds quarantine threshold → `QUARANTINE`. |
| **Sign-Flip Attack (Mature Rounds)** | Attacker negates gradient direction to push model away from minimum in rounds $t \ge 3$. | **Cosine Direction Check**: Negative cosine similarity ($< -0.1$) against prior trajectory $\Delta G_{t-1} \rightarrow$ `FLAG`. |
| **Weight Inflation via Sample Spoofing** | Hospital artificially inflates reported $N_i$ between rounds to gain higher FedAvg weight. | **Sample-Count Consistency Check**: Compares $N_i$ to previous rounds; $\ge 3\times$ or $\le 0.33\times$ change $\rightarrow$ `QUARANTINE`. |
| **Moderate Outliers in Aggregation** | Subtle outliers that evade quarantine or updates marked `FLAG`. | **Coordinate-wise Median & Trimmed Mean**: Replaces linear averaging with robust statistics ($K \ge 3$). |
| **Silent Verification Failure** | Corrupt database, missing base checkpoint, or dimension mismatch causing silent bypass. | **Auditable Skipped Checks**: All skipped checks recorded in `metrics["checks_skipped"]` and exposed in UI (updates still accepted; not fail-closed). |

### Threats NOT Covered (Out of Scope / Residual Risk)

1. **Adaptive Attacks Inside Clean Bounds**: An attacker who knows the calibrated threshold (e.g. ratio $\le 2.2$) can craft malicious updates that remain below the flag threshold while slowly biasing features.
2. **Colluding Majority ($> 50\%$ Byzantine)**: Coordinate-wise median has a theoretical breakdown point of $50\%$. If $\ge 50\%$ of participating nodes collude, robust aggregation cannot guarantee safety. Trimmed mean breakdown point is $\beta$ (default $\beta = 0.2$, tolerating at most $20\%$ outliers).
3. **Clean-Looking Data Poisoning / Backdoors**: Backdoors or dirty-label attacks trained on biased local data that exhibit natural loss and normal gradient magnitudes cannot be detected by parameter norm or direction tests alone without central validation data.
4. **Sign-Flip Attacks in Rounds 1–2**: Early rounds have no established global update trajectory ($G_{t-1} - G_{t-2}$). Directional verification is mathematically undefined and inactive in rounds 1 and 2.
5. **Robust Aggregation with 2 Hospitals**: Coordinate-wise median and trimmed mean mathematically require at least $K \ge 3$ participants. With 2 hospitals, aggregation automatically falls back to weighted FedAvg.

---

## Defense Mechanisms & Mathematical Formulations

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Hospital Update Intake                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Parameter Sanity: Any NaN or Inf in tensors?                        │
│    ├── Yes ──────────────────────────────────────────────► REJECT     │
│    └── No                                                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Sample-Count Consistency: Ni > 0 and consistent with prior rounds?  │
│    ├── Ni <= 0 or Ni > 10M ──────────────────────────────► REJECT/FLAG │
│    ├── |Ni_curr / Ni_prev| >= 3.0 or <= 0.33 ────────────► QUARANTINE │
│    └── Normal Ni                                                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Trainable L2 Delta Norm Bound:                                      │
│    Δ = W_trainable - B_trainable                                       │
│    ratio = ||Δ||₂ / (||B_trainable||₂ + 1e-12)                         │
│    ├── ratio > Quarantine Threshold (5.5x Non-DP, 1.0x DP) ► QUARANTINE│
│    ├── ratio > Flag Threshold (2.2x Non-DP, 0.4x DP) ────► FLAG        │
│    └── Normal ratio                                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Cosine Direction Check (Rounds >= 3):                               │
│    cos_sim = (Δ_curr · Δ_G_prev) / (||Δ_curr||₂ * ||Δ_G_prev||₂)       │
│    ├── cos_sim < -0.1 ───────────────────────────────────► FLAG        │
│    └── cos_sim >= -0.1 (or round < 3 recorded as skipped)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Aggregation Tier:                                                   │
│    - ACCEPTED & FLAGGED updates pass to Aggregator                     │
│    - QUARANTINED & REJECTED updates are excluded                       │
│    - Strategy: FedAvg | FedProx | Coordinate Median | Trimmed Mean     │
│    - BatchNorm Running Statistics averaged over accepted clients only  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Parameter Sanity (NaN / Inf Check)
For each parameter array $P_k \in \text{handoff.parameters}$:
$$\text{has\_nan\_inf} = \bigvee_{k} \left( \exists x \in P_k : x = \text{NaN} \lor |x| = \infty \right)$$
If true, the update is immediately rejected with verdict `REJECT`.

### 2. Sample-Count Consistency Check
Compares the current round's sample count $N_{\text{curr}}$ against the most recent accepted round $N_{\text{prev}}$ for the same participant:
$$\text{ratio} = \frac{N_{\text{curr}}}{N_{\text{prev}}}$$
- If $N_{\text{curr}} \le 0$: `REJECT`
- If $N_{\text{curr}} > 10,000,000$: `FLAG`
- If $\text{ratio} \ge 3.0$ or $\text{ratio} \le 0.333$: `QUARANTINE`
- If $N_{\text{curr}} \ne N_{\text{prev}}$: `FLAG`

### 3. L2 Parameter Delta Norm Bound (Trainable Weights Only)
Previous formulations included BatchNorm running buffers (`running_mean`, `running_var`) and integer counters (`num_batches_tracked`), which drift nonlinearly during local training. The Module 12 check filters parameter arrays strictly to **trainable weights** using architecture parameter names:

$$\Delta = \bigoplus_{i \in \text{trainable}} \text{vec}(P_i - B_i), \quad B = \bigoplus_{i \in \text{trainable}} \text{vec}(B_i)$$
$$\text{ratio} = \frac{\|\Delta\|_2}{\|B\|_2 + 10^{-12}}$$

- **Non-DP default**: Flag at `ratio > 2.2`, Quarantine at `ratio > 5.5`
- **DP default**: Flag at `ratio > 0.4`, Quarantine at `ratio > 1.0`
- Both use the **exact same documented safety margin** (2.0x for Flag, 5.0x for Quarantine) over the maximum clean ratio observed in real historical database runs.

### 4. Cosine Direction Check (Rounds $t \ge 3$)
When two previous global models $G_{t-1}$ and $G_{t-2}$ exist for the job:
$$\Delta G_{\text{prev}} = \text{vec}(G_{t-1}) - \text{vec}(G_{t-2})$$
$$\cos(\Delta, \Delta G_{\text{prev}}) = \frac{\Delta \cdot \Delta G_{\text{prev}}}{\|\Delta\|_2 \|\Delta G_{\text{prev}}\|_2}$$
If $\cos(\Delta, \Delta G_{\text{prev}}) < -0.1$ (or `m12_min_cosine_similarity`): verdict is `FLAG`.
If round $< 3$ or global models are missing: check is skipped and recorded in `metrics["checks_skipped"]["cosine"]`.

### 5. Coordinate-wise Median
For parameter coordinate $j$ across $K$ accepted clients:
$$W_{\text{global}, j} = \text{median}\left(\{W_{1, j}, W_{2, j}, \dots, W_{K, j}\}\right)$$
Breakdown point is $50\%$.

### 6. Trimmed Mean
For trim ratio $\beta \in (0, 0.5)$ and trim count $m = \lfloor K \cdot \beta \rfloor$:
$$W_{\text{global}, j} = \frac{1}{K - 2m} \sum_{r = m+1}^{K - m} W_{(r), j}$$
where $W_{(r), j}$ is the $r$-th order statistic at coordinate $j$.

---

## Empirical Calibration Table (Real Training Data)

Calibrated via read-only inspection (`mode=ro`) of all 19 clean accepted updates and 10 global models
in `federation_metadata.db` and `federation_artifacts/`. All metrics are computed strictly over
**trainable weights only**:

| Job ID | Round | Participant | Architecture | Norm Type | $\|\Delta\|_2$ | $\|B\|_2$ | Clean Ratio |
|---|---|---|---|---|---|---|---|
| `JOB_MAL_RES18_CLS2_V1` | R1 | HOSP_123456 | resnet18 | batch | 62.5344 | 111.6973 | 0.5599 |
| `JOB_MAL_RES18_CLS2_V1` | R1 | HOSP_868720 | resnet18 | batch | 40.9434 | 111.6973 | 0.3666 |
| `JOB_SKINCAN_MNV2_CLS2_V1` | R1 | HOSP_868720 | mobilenet_v2 | batch | 63.7514 | 185.3405 | 0.3440 |
| `JOB_SKINCAN_MNV2_CLS2_V1` | R1 | HOSP_123456 | mobilenet_v2 | batch | 14.7536 | 185.3405 | 0.0796 |
| `JOB_MAL_MNV2_CLS2_1` | R1 | HOSP_999999 | mobilenet_v2 | batch | 67.9351 | 185.4506 | 0.3663 |
| `JOB_MAL_MNV2_CLS2_1` | R1 | HOSP_888888 | mobilenet_v2 | batch | 42.1301 | 185.4506 | 0.2272 |
| `JOB_MAL_MNV2_CLS2_1` | R2 | HOSP_999999 | mobilenet_v2 | batch | 77.0031 | 189.2471 | 0.4069 |
| `JOB_MAL_MNV2_CLS2_1` | R2 | HOSP_888888 | mobilenet_v2 | batch | 40.6853 | 189.2471 | 0.2150 |
| `JOB_MAL_MNV2_CLS2_1` | R3 | HOSP_888888 | mobilenet_v2 | batch | 37.3089 | 203.3662 | 0.1835 |
| `JOB_MAL_MNV2_CLS2_1` | R3 | HOSP_999999 | mobilenet_v2 | batch | 61.1077 | 203.3662 | 0.3005 |
| `JOB_MAL_RES18_CLS2_V2` | R1 | HOSP_888888 | resnet18 | batch | 42.5503 | 111.7188 | 0.3809 |
| `JOB_MAL_RES18_CLS2_V2` | R1 | HOSP_999999 | resnet18 | batch | 54.9619 | 111.7188 | 0.4920 |
| `JOB_MAL_RES18_CLS2_V2` | R2 | HOSP_999999 | resnet18 | batch | 131.3501 | 119.0699 | **1.1031** |
| `JOB_MAL_RES18_CLS2_V2` | R2 | HOSP_888888 | resnet18 | batch | 87.1786 | 119.0699 | 0.7322 |
| `JOB_MAL_MNV3S_CLS2_V2` (DP) | R1 | HOSP_888888 | mobilenet_v3_small | group | 14.4606 | 134.0226 | 0.1079 |
| `JOB_MAL_MNV3_CLS2_V2` (DP) | R1 | HOSP_888888 | mobilenet_v3_small | group | 24.3638 | 134.2230 | 0.1815 |
| `JOB_MAL_MNV3_CLS2_V2` (DP) | R1 | HOSP_999999 | mobilenet_v3_small | group | 26.0957 | 134.2230 | **0.1944** |
| `JOB_MAL_MNV3_CLS2_V2` (DP) | R2 | HOSP_888888 | mobilenet_v3_small | group | 22.9790 | 135.7914 | 0.1692 |
| `JOB_MAL_MNV3_CLS2_V2` (DP) | R2 | HOSP_999999 | mobilenet_v3_small | group | 16.7875 | 135.7914 | 0.1236 |

### Summary & Threshold Calibration

| Category | Count | Min Ratio | Max Clean Ratio | Mean Ratio | Flag Threshold ($2.0\times$ Max) | Quarantine Threshold ($5.0\times$ Max) |
|---|---|---|---|---|---|---|
| **Non-DP** | 14 | 0.0796 | **1.1031** | 0.4112 | **2.2** | **5.5** |
| **DP** | 5 | 0.1079 | **0.1944** | 0.1553 | **0.4** | **1.0** |

Both DP and Non-DP thresholds enforce the exact same documented margin:
- **Flag Margin**: $2.0\times$ observed maximum clean ratio (100% safety buffer over normal SGD drift).
- **Quarantine Margin**: $5.0\times$ observed maximum clean ratio (400% safety buffer).

### Per-Job Calibration Behind 2.2x / 5.5x Thresholds

| Job ID | DP Enabled | Architecture | Clean Updates | Max Trainable Ratio | Flag Margin ($2.0\times$) | Quarantine Margin ($5.0\times$) |
|---|---|---|---|---|---|---|
| `JOB_MAL_RES18_CLS2_V1` | No | resnet18 | 2 | 0.5599 | — | — |
| `JOB_SKINCAN_MNV2_CLS2_V1` | No | mobilenet_v2 | 2 | 0.3440 | — | — |
| `JOB_MAL_MNV2_CLS2_1` | No | mobilenet_v2 | 6 | 0.4069 | — | — |
| `JOB_MAL_RES18_CLS2_V2` | No | resnet18 | 4 | **1.1031** | **2.2** | **5.5** |
| `JOB_MAL_MNV3S_CLS2_V2` | Yes | mobilenet_v3_small | 1 | 0.1079 | — | — |
| `JOB_MAL_MNV3_CLS2_V2` | Yes | mobilenet_v3_small | 4 | **0.1944** | **0.4** | **1.0** |
| **All Non-DP Jobs** | No | resnet18, mobilenet_v2 | **14** | **1.1031** | **2.2** ($2.0\times$ margin) | **5.5** ($5.0\times$ margin) |
| **All DP Jobs** | Yes | mobilenet_v3_small | **5** | **0.1944** | **0.4** ($2.0\times$ margin) | **1.0** ($5.0\times$ margin) |

---

## Evaluation Benchmark (`federation/eval_m12.py`)

Evaluation benchmark executed with command `python federation/eval_m12.py` under controlled, reproducible conditions:
- **Command**: `python federation/eval_m12.py`
- **Seeds**: `torch.manual_seed(42)`, `np.random.seed(42)`
- **Clients**: 5 synthetic clients ($K = 5$), dimension = 2,000 parameters.
- **Client Distribution**: 4 honest clients with noisy gradients pointing along true optimization objective; 1 attacker executing various model poisoning strategies.
- **Output Artifacts**: Results written to `outputs/m12_evaluation_summary.json` and `outputs/m12_evaluation_summary.txt` (both gitignored under `outputs/`).

| Scenario | Attacker Ratio | Hook Verdict | Naive FedAvg L2 Err | FedAvg + M12 L2 Err | Coord Median L2 Err | Trimmed Mean L2 Err |
|---|---|---|---|---|---|---|
| **No Attack (5 Honest)** | 0.51 | `ACCEPT` | 0.0000 | 0.0000 | 0.6606 | 0.3689 |
| **Extreme Scaling (12x)** | 6.11 | `QUARANTINE` | 5.4161 | **0.4965** | 1.1777 | 1.0453 |
| **Moderate Scaling (6x)** | 3.06 | `FLAG` | 2.7305 | 2.7305 | **1.1696** | **1.0294** |
| **Sign-Flip Attack (-1.5x)** | 0.76 | `ACCEPT` | 0.8199 | 0.8199 | 0.9666 | **0.7734** |
| **Gaussian Noise Attack** | 51.30 | `QUARANTINE` | 45.3805 | **0.4965** | 1.0972 | 0.9752 |

> **Understanding the Benchmark Metrics & Columns**:
> - **L2 Error Metric (Columns 4–7)**: Each number in the aggregator columns measures the Euclidean distance (L2 norm) between the aggregated update vector and the ground-truth mean of the honest clients: $\|\Delta_{\text{agg}} - \Delta_{\text{honest\_consensus}}\|_2$, computed on synthetic 2,000-dimensional parameter vectors. **Lower is better** (a score of `0.0000` indicates perfect alignment with the honest consensus).
> - **Parameter-Space Simulation (Not Accuracy)**: This benchmark is a controlled geometric parameter-space simulation that quantifies how effectively aggregation algorithms and intake filtering suppress malicious vector perturbations. It measures parameter-vector distortion, **not model classification accuracy or validation loss**.
> - **"Attacker Ratio" Column**: Measures $\|\Delta_5\|_2 / (\|B\|_2 + 10^{-12})$, the L2 ratio of Client 5's parameter update relative to the base model:
>   - In the **No Attack (5 Honest)** row, Client 5 is a legitimate, benign hospital node; its ratio of **0.51** represents normal, clean local SGD training drift (consistent with real database ratios of 0.08–1.10).
>   - In the attack rows, Client 5 executes a poisoned update; the ratio reflects the scaling factor or noise magnitude injected (e.g., 6.11 under 12x scaling, 51.30 under Gaussian noise).

### Key Benchmark Observations:
1. **Quarantine Defense**: Under extreme scaling (12x) and Gaussian noise, the M12 hook quarantines the malicious update, reducing aggregation error by **90.8%** and **98.9%** respectively.
2. **Robust Aggregation Synergy**: When an attack is moderately scaled (6x, ratio 3.06) and falls between flag and quarantine thresholds, Coordinate-wise Median and Trimmed Mean reduce error from 2.7305 down to ~1.03–1.16.
3. **Sign-Flip Defense**: In round 1 before trajectory is established, Trimmed Mean achieves the lowest error (0.7734) among all aggregators.

---

## Configuration Keys

Configured in `training_requirements` or `aggregation_configuration`:

| Key | Location | Type | Default | Description |
|---|---|---|---|---|
| `strategy` | `aggregation_configuration` | string | `"fedavg"` | Aggregation strategy: `"fedavg"`, `"fedprox"`, `"median"`, `"trimmed_mean"`. |
| `trim_ratio` | `aggregation_configuration` | float | `0.2` | Fraction of updates to trim from each tail in trimmed mean ($0 < \beta < 0.5$). |
| `m12_max_l2_ratio` | `training_requirements` | float | Non-DP: 2.2, DP: 0.4 | Explicit override for delta-to-base ratio flag threshold. |
| `m12_quarantine_l2_ratio` | `training_requirements` | float | Non-DP: 5.5, DP: 1.0 | Explicit override for delta-to-base ratio quarantine threshold. |
| `m12_max_l2_norm` | `training_requirements` | float | None | Explicit absolute L2 norm flag threshold $\|\Delta\|_2$. |
| `m12_quarantine_l2_norm` | `training_requirements` | float | None | Explicit absolute L2 norm quarantine threshold $\|\Delta\|_2$. |
| `m12_min_cosine_similarity` | `training_requirements` | float | `-0.1` | Cosine similarity lower bound with prior trajectory (rounds $\ge 3$). |

---

## Verification

### Automated Test Suites

Execute the test suites locally with:

```bash
# Module 12 Security Test Suite (19 tests)
python -m pytest federation/tests/test_m12_security.py -v

# Full Federation Test Suite (139 tests, including Flower tests)
python -m pytest federation/tests -q

# Hospital Client Pipeline Suite (860 test cases)
python -m pytest hospital_client -q

# Backend API & Federation Test Suite (183 tests)
cd backend && npm test

# Frontend Production Build Check
cd frontend && npx vite build
```

### Verified Test Results:
- `federation/tests/test_m12_security.py`: **19 passed, 0 failed** (14.51s)
- `federation/tests`: **139 passed, 0 failed** (including full Flower test suite, 436.60s)
- `hospital_client`: **858 passed, 2 skipped** (GPU hardware skips; 0 failed)
- `backend` (`npm test`): **183 passed, 0 failed** (4 test suites, 66.22s)
- `frontend` (`npx vite build`): **built successfully** (1694 modules transformed, 0 errors)

---

## Known Limitations

1. **Two-Hospital Limitation**: When only 2 hospitals participate in a round, robust aggregation (median / trimmed mean) is mathematically undefined and automatically falls back to weighted FedAvg. Defense in 2-hospital runs relies entirely on the M12 intake security hook.
2. **Early Round Direction**: Rounds 1 and 2 lack a previous global trajectory. Directional sign-flip detection is skipped by design and recorded in `checks_skipped`.
3. **Adaptive Attack Surface**: An attacker with knowledge of the local base model can craft subtle adversarial updates with norm ratio $\le 2.2$ that evade intake quarantine.
4. **Skipped Checks Acceptance Policy (Not Fail-Closed)**: When a check cannot execute (e.g., base model checkpoint unavailable, database connectivity issue, fewer than 2 previous global models, or parameter count mismatch), the check is skipped and the reason is recorded in `metrics["checks_skipped"]` and displayed in the UI. The update is **still accepted**. This behavior prevents the federation from deadlocking when historical models or baseline checkpoints are missing, but it means skipped checks operate in a fail-open (audited) manner rather than fail-closed. Only parameter corruption (`NaN`/`±Inf`) and explicit threshold violations reject or quarantine an update.
5. **Sign-Flip Attack Acceptance & Median Degradation**: The -1.5x sign-flip attack was **ACCEPTED** by the hook because its ratio (0.76) falls well inside the clean bound (2.2x), and round 1 lacks an established trajectory for cosine checks. Furthermore, in this simulation, Coordinate-wise Median performed *worse* than naive FedAvg against the sign-flip attack (L2 error 0.9666 vs. 0.8199), whereas Trimmed Mean provided the best defense (0.7734).
6. **FLAG Verdict Is An Alert Only (Non-Blocking)**: Flagged updates are still accepted and included in aggregation. As observed in the 6x attack row (ratio 3.06), `FedAvg + M12` equals naive FedAvg (error 2.7305). Protection against flagged updates comes solely from `QUARANTINE` or server-side robust aggregation (Coordinate-wise Median at 1.1696 or Trimmed Mean at 1.0294), both of which require at least 3 hospitals ($K \ge 3$).
7. **Cost of Robust Aggregation Without an Attack**: In the clean baseline scenario without an attack (5 honest hospitals), standard FedAvg achieves optimal consensus (error 0.0000). Applying Coordinate-wise Median or Trimmed Mean moves the aggregate away from the honest mean (errors of 0.6606 and 0.3689 respectively), which is the inherent accuracy/precision cost paid for Byzantine robustness during benign training rounds.

---

## Future Work

1. **Dynamic Krum / Multi-Krum Aggregation**: Add Krum Byzantine defense for settings with $\ge 4$ participants to select the candidate update minimizing sum of squared distances to neighbors.
2. **Local Validation Loss Auditing**: Allow the central server to evaluate submitted updates against an encrypted validation set held in a secure enclave before acceptance.
3. **Adaptive Threshold Adjustment**: Dynamically tighten L2 ratio bounds as global rounds progress and convergence slows ($\Delta_t \rightarrow 0$).
