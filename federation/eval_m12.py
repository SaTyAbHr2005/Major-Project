"""
Module 12: Byzantine Defense and Robust Aggregation Evaluation Script.

Evaluates defenses against adversarial model updates under controlled, reproducible conditions:
- Fixed seeds (torch.manual_seed(42), np.random.seed(42))
- 5 synthetic clients (100 samples each):
  * Baseline: 5 honest clients
  * Scaling Attack: 4 honest clients, 1 attacker (10x scaling)
  * Sign-Flip Attack: 4 honest clients, 1 attacker (-1x sign flip)
  * Random Noise Attack: 4 honest clients, 1 attacker (Gaussian noise N(0, 10))
- Defenses evaluated:
  1. Naive FedAvg (No M12 defense)
  2. FedAvg + M12 Security Hook (Hook filters / quarantines / rejects malicious updates)
  3. Coordinate-wise Median
  4. Trimmed Mean (trim_ratio = 0.2)
- Metrics:
  * L2 error from honest consensus: ||Delta_agg - Delta_honest_consensus||_2
  * Cosine similarity with honest consensus: cos(Delta_agg, Delta_honest_consensus)
  * Attack detection verdict from M12 hook

Writes full results to outputs/m12_evaluation_summary.json and outputs/m12_evaluation_summary.txt.
"""
import os
import sys
import json
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from federation.security_hooks import M12SecurityHook
from federation.models import FederationJob, FederationRound, FederationUpdate, UpdateStatus
from hospital_client.training.federation import build_federation_handoff


def aggregate_weighted_fedavg(client_updates_tuples):
    total_samples = sum(w for _, w in client_updates_tuples)
    return [sum((w / total_samples) * p[0] for p, w in client_updates_tuples)]


def aggregate_coordinate_median(client_updates_tuples):
    stacked = np.stack([p[0] for p, w in client_updates_tuples], axis=0)
    return [np.median(stacked, axis=0)]


def aggregate_trimmed_mean(client_updates_tuples, trim_ratio=0.2):
    k = len(client_updates_tuples)
    trim_count = max(1, int(k * trim_ratio))
    if 2 * trim_count >= k:
        trim_count = max(0, (k - 1) // 2)
    stacked = np.stack([p[0] for p, w in client_updates_tuples], axis=0)
    sorted_stacked = np.sort(stacked, axis=0)
    trimmed = sorted_stacked[trim_count : k - trim_count]
    return [np.mean(trimmed, axis=0)]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_synthetic_data(dim=1000):
    """
    Creates base model and honest local updates pointing towards an optimal direction.
    """
    base_params = [np.random.randn(dim).astype(np.float32) * 0.1]
    true_direction = np.random.randn(dim).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    # 4 honest clients with slightly noisy gradients in the true direction
    honest_updates = []
    for i in range(4):
        noise = np.random.randn(dim).astype(np.float32) * 0.05
        delta = (true_direction * 0.2 + noise).astype(np.float32)
        param = (base_params[0] + delta).astype(np.float32)
        honest_updates.append(param)

    # A 5th honest update for the 5-honest baseline
    noise_5 = np.random.randn(dim).astype(np.float32) * 0.05
    honest_5th = (base_params[0] + (true_direction * 0.2 + noise_5)).astype(np.float32)

    return base_params, honest_updates, honest_5th, true_direction


def run_evaluation():
    set_seed(42)
    dim = 2000
    base_params, honest_4, honest_5th, true_dir = create_synthetic_data(dim=dim)

    # Compute honest consensus ground truth
    all_honest = honest_4 + [honest_5th]
    honest_deltas = [p - base_params[0] for p in all_honest]
    honest_consensus_delta = np.mean(honest_deltas, axis=0)
    honest_consensus_norm = np.linalg.norm(honest_consensus_delta)

    # Construct attack updates for Client 5
    client4_delta = honest_4[0] - base_params[0]
    # 1. Extreme Scaling Attack (12x, ratio > 5.5 quarantine threshold)
    attack_scaling_12x = (base_params[0] + client4_delta * 12.0).astype(np.float32)

    # 2. Moderate Scaling Attack (6x, ratio > 2.2 flag threshold)
    attack_scaling_6x = (base_params[0] + client4_delta * 6.0).astype(np.float32)

    # 3. Sign-Flip Attack (-1.5x)
    attack_sign_flip = (base_params[0] - client4_delta * 1.5).astype(np.float32)

    # 4. Gaussian Noise Attack
    attack_noise = (base_params[0] + np.random.randn(dim).astype(np.float32) * 5.0).astype(np.float32)

    scenarios = [
        ("No Attack (5 Honest)", all_honest, "None"),
        ("Extreme Scaling (12x)", honest_4 + [attack_scaling_12x], "Scaling (12x)"),
        ("Moderate Scaling (6x)", honest_4 + [attack_scaling_6x], "Scaling (6x)"),
        ("Sign-Flip Attack (-1.5x)", honest_4 + [attack_sign_flip], "Sign Flip (-1.5x)"),
        ("Gaussian Noise Attack", honest_4 + [attack_noise], "Gaussian Noise (sigma=5.0)"),
    ]

    job = FederationJob(
        federation_job_id="EVAL_M12",
        task="malaria",
        task_type="image_classification",
        architecture="synthetic_linear",
        num_classes=2,
        class_mapping={"0": 0, "1": 1},
        expected_participants=[f"H{i}" for i in range(1, 6)],
        minimum_participants=5,
        deadline="",
        training_requirements={"m12_max_l2_ratio": 2.2, "m12_quarantine_l2_ratio": 5.5},
        aggregation_configuration={},
    )
    rnd = FederationRound(
        round_id="R1",
        federation_job_id="EVAL_M12",
        round_number=1,
        expected_participants=[f"H{i}" for i in range(1, 6)],
        minimum_participants=5,
        deadline="",
        base_model_id="BASE_SYNTH",
        base_model_version=1,
        base_model_checksum="dummy",
    )

    hook = M12SecurityHook(storage=None)

    def evaluate_hook_on_params(param_array):
        handoff = build_federation_handoff(
            parameters=[param_array],
            model_id="test",
            model_version=1,
            architecture="synthetic_linear",
            base_model_id="BASE_SYNTH",
            base_model_version=1,
            base_model_checksum="dummy",
            num_train_samples=100,
            class_mapping={"0": 0, "1": 1},
            training_metrics={"loss": 0.3},
            validation_metrics={},
            training_configuration={},
            device="cpu",
            precision="fp32",
            status="TRAINING_COMPLETED_AWAITING_FEDERATION",
        )
        u = FederationUpdate(
            update_id="u_eval",
            federation_job_id="EVAL_M12",
            round_id="R1",
            participant_id="H5",
            model_id="test",
            architecture="synthetic_linear",
            artifact_location="",
            artifact_checksum="",
            parameter_count=dim,
            parameter_metadata={},
            num_train_samples=100,
            training_configuration={},
        )
        # Mock load_base_parameters for synthetic eval
        old_loader = sys.modules["federation.security_hooks"]._load_base_parameters
        sys.modules["federation.security_hooks"]._load_base_parameters = lambda s, j, r: (base_params, None)
        try:
            decision = hook.evaluate_update(u, handoff, job=job, rnd=rnd)
        finally:
            sys.modules["federation.security_hooks"]._load_base_parameters = old_loader
        return decision

    results_table = []
    json_output = {"scenarios": []}

    for scenario_name, client_params, attack_type in scenarios:
        # Check client 5 with hook
        decision_c5 = evaluate_hook_on_params(client_params[4])
        c5_verdict = decision_c5.verdict
        c5_ratio = decision_c5.metrics.get("delta_to_base_ratio", 0.0)

        # 1. Naive FedAvg (all 5 clients)
        client_updates_tuples = [([p], 100) for p in client_params]
        agg_naive = aggregate_weighted_fedavg(client_updates_tuples)[0]
        delta_naive = agg_naive - base_params[0]
        err_naive = float(np.linalg.norm(delta_naive - honest_consensus_delta))
        cos_naive = float(np.dot(delta_naive, honest_consensus_delta) / (np.linalg.norm(delta_naive) * honest_consensus_norm + 1e-12))

        # 2. FedAvg + M12 Hook (Excludes quarantined / rejected updates)
        accepted_tuples = []
        for i, p in enumerate(client_params):
            dec = evaluate_hook_on_params(p)
            if dec.verdict in ("ACCEPT", "FLAG"):
                accepted_tuples.append(([p], 100))
        if accepted_tuples:
            agg_m12 = aggregate_weighted_fedavg(accepted_tuples)[0]
        else:
            agg_m12 = base_params[0]
        delta_m12 = agg_m12 - base_params[0]
        err_m12 = float(np.linalg.norm(delta_m12 - honest_consensus_delta))
        cos_m12 = float(np.dot(delta_m12, honest_consensus_delta) / (np.linalg.norm(delta_m12) * honest_consensus_norm + 1e-12))

        # 3. Coordinate-wise Median
        agg_median = aggregate_coordinate_median(client_updates_tuples)[0]
        delta_median = agg_median - base_params[0]
        err_median = float(np.linalg.norm(delta_median - honest_consensus_delta))
        cos_median = float(np.dot(delta_median, honest_consensus_delta) / (np.linalg.norm(delta_median) * honest_consensus_norm + 1e-12))

        # 4. Trimmed Mean (trim_ratio=0.2)
        agg_trimmed = aggregate_trimmed_mean(client_updates_tuples, trim_ratio=0.2)[0]
        delta_trimmed = agg_trimmed - base_params[0]
        err_trimmed = float(np.linalg.norm(delta_trimmed - honest_consensus_delta))
        cos_trimmed = float(np.dot(delta_trimmed, honest_consensus_delta) / (np.linalg.norm(delta_trimmed) * honest_consensus_norm + 1e-12))

        entry = {
            "scenario": scenario_name,
            "attack_type": attack_type,
            "attacker_delta_ratio": round(c5_ratio, 4),
            "hook_verdict": c5_verdict,
            "naive_fedavg": {"l2_error": round(err_naive, 4), "cosine": round(cos_naive, 4)},
            "m12_hook_fedavg": {"l2_error": round(err_m12, 4), "cosine": round(cos_m12, 4)},
            "coord_median": {"l2_error": round(err_median, 4), "cosine": round(cos_median, 4)},
            "trimmed_mean": {"l2_error": round(err_trimmed, 4), "cosine": round(cos_trimmed, 4)},
        }
        json_output["scenarios"].append(entry)

        results_table.append({
            "Scenario": scenario_name,
            "Attacker Ratio": f"{c5_ratio:.2f}",
            "Hook Verdict": c5_verdict,
            "Naive FedAvg L2 Err": f"{err_naive:.4f}",
            "FedAvg+M12 L2 Err": f"{err_m12:.4f}",
            "Median L2 Err": f"{err_median:.4f}",
            "Trimmed Mean L2 Err": f"{err_trimmed:.4f}",
            "FedAvg+M12 Cosine": f"{cos_m12:.4f}",
        })

    # Save to outputs/
    outputs_dir = os.path.join(REPO_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, "m12_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    txt_path = os.path.join(outputs_dir, "m12_evaluation_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== Module 12 Byzantine Defense & Robust Aggregation Evaluation ===\n\n")
        headers = ["Scenario", "Attacker Ratio", "Hook Verdict", "Naive FedAvg L2", "FedAvg+M12 L2", "Median L2", "Trimmed Mean L2", "FedAvg+M12 Cos"]
        col_w = [26, 16, 14, 18, 16, 12, 18, 16]
        hdr_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
        f.write(hdr_line + "\n")
        f.write("-+-".join("-" * w for w in col_w) + "\n")
        for r in results_table:
            vals = [r["Scenario"], r["Attacker Ratio"], r["Hook Verdict"], r["Naive FedAvg L2 Err"], r["FedAvg+M12 L2 Err"], r["Median L2 Err"], r["Trimmed Mean L2 Err"], r["FedAvg+M12 Cosine"]]
            f.write(" | ".join(v.ljust(w) for v, w in zip(vals, col_w)) + "\n")

    print("\n" + "=" * 95)
    print("MODULE 12 EVALUATION BENCHMARK RESULTS (outputs/m12_evaluation_summary.json)")
    print("=" * 95)
    print(f"| {'Scenario':26s} | {'Attacker Ratio':14s} | {'Hook Verdict':12s} | {'Naive FedAvg':12s} | {'FedAvg+M12':12s} | {'Coord Median':12s} | {'Trimmed Mean':12s} |")
    print(f"|{'-'*28}|{'-'*16}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|")
    for r in results_table:
        print(f"| {r['Scenario']:26s} | {r['Attacker Ratio']:14s} | {r['Hook Verdict']:12s} | {r['Naive FedAvg L2 Err']:12s} | {r['FedAvg+M12 L2 Err']:12s} | {r['Median L2 Err']:12s} | {r['Trimmed Mean L2 Err']:12s} |")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_evaluation()
