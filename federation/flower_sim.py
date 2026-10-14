"""
Flower simulation bench (Module 9): a real Flower server and N simulated hospital clients
on one machine, for federated experiments (centralised vs federated, IID vs Non-IID,
FedAvg vs FedProx, communication cost).

    python -m federation.flower_sim --dataset <Module 5 output dir> --output <results dir> \
        --clients 3 --rounds 5 --partition label_skew --alpha 0.5 \
        --strategy fedprox --mu 0.01 --architecture resnet18 --epochs 1 --batch-size 32 --seed 42

How it runs
- The Module 5 train and validation manifests are split among the clients (IID, or label skew
  with a Dirichlet(alpha) share of every class per client). Every record goes to exactly one
  client; the per-client manifests only reference the original files (nothing is copied).
- The server is Flower's gRPC server (`flwr.server.start_server`) with Flower's own FedAvg or
  FedProx strategy. Each client is a separate process using `flwr.client.start_client`. They talk
  over gRPC on 127.0.0.1, so model parameters really go through Flower's transport.
- A client trains with the unchanged Module 7 Trainer, starting from the global model of the
  round (loaded through the same checksum-verified base-model path as a real hospital). FedProx's
  proximal_mu reaches the client in the fit config and becomes TrainingConfig.proximal_mu.
- After every round the server evaluates the global model on the Module 5 *test* manifest
  (round 0 = the untrained starting model). Test data is never used for training or selection.
- Clients train one at a time (a shared lock), so several clients fit on a small GPU.

Results: <output>/results.json (full configuration, partition class counts, per-round metrics,
client metrics, durations, bytes exchanged) and <output>/rounds.csv.

Limitations
- Records are split individually; patient/group identifiers are not kept together across clients.
- Flower's FedAvg averages every array, including BatchNorm's integer counters (cast back to int);
  the application protocol in federation/aggregation.py copies those from the first participant.
- The number of rounds is fixed; there is no early stopping in the simulation.
- `start_server`/`start_client` are deprecated since Flower 1.13 in favour of SuperLink/SuperNode,
  but they need no extra services (and no Ray), which keeps the bench a single command.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

SIM_VERSION = "1.0"
PARTITIONS = ("iid", "label_skew")
STRATEGIES = ("fedavg", "fedprox")


# ── Partitioning (pure, tested without Flower) ─────────────────────────────────────────────

def partition_records(records: List[dict], n_clients: int, mode: str, alpha: float, seed: int) -> List[List[dict]]:
    """Split records among clients; every record lands in exactly one client.

    iid:        a seeded shuffle cut into near-equal parts.
    label_skew: for every class, client shares drawn from Dirichlet(alpha); small alpha = more skew.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    rng = np.random.default_rng(seed)
    if mode == "iid":
        order = rng.permutation(len(records))
        return [[records[i] for i in sorted(part)] for part in np.array_split(order, n_clients)]
    if mode != "label_skew":
        raise ValueError(f"partition must be one of {list(PARTITIONS)}, got {mode!r}")
    if not alpha > 0:
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    by_label: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_label.setdefault(r["label"], []).append(i)
    parts: List[List[int]] = [[] for _ in range(n_clients)]
    for label in sorted(by_label):
        idx = rng.permutation(by_label[label])
        shares = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
        for c, chunk in enumerate(np.split(idx, cuts)):
            parts[c].extend(chunk.tolist())
    return [[records[i] for i in sorted(part)] for part in parts]


def class_counts(records: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return dict(sorted(counts.items()))


def write_client_datasets(dataset_dir: str, out_dir: str, n_clients: int, mode: str, alpha: float, seed: int) -> List[Dict[str, Any]]:
    """Write client_<k>/{train,validation}_manifest.json under out_dir. Returns one summary per client."""
    from hospital_client.training.dataset import load_manifest
    train = load_manifest(os.path.join(dataset_dir, "train_manifest.json"))
    val_path = os.path.join(dataset_dir, "validation_manifest.json")
    val = load_manifest(val_path) if os.path.exists(val_path) else {"records": []}

    train_parts = partition_records(train["records"], n_clients, mode, alpha, seed)
    val_parts = partition_records(val["records"], n_clients, mode, alpha, seed + 1)
    empty = [k for k, part in enumerate(train_parts) if not part]
    if empty:
        raise ValueError(f"Client(s) {empty} received no training images. Use fewer clients or a larger alpha.")

    summaries = []
    for k in range(n_clients):
        d = os.path.join(out_dir, f"client_{k}")
        os.makedirs(d, exist_ok=True)
        for name, manifest, part in (("train", train, train_parts[k]), ("validation", val, val_parts[k])):
            if name == "validation" and not part:
                continue
            with open(os.path.join(d, f"{name}_manifest.json"), "w") as f:
                json.dump({**manifest, "records": part, "total_records": len(part)}, f)
        summaries.append({"client": k, "dataset_dir": d,
                          "train": class_counts(train_parts[k]), "validation": class_counts(val_parts[k])})
    return summaries


# ── Model helpers (Module 6/7) ─────────────────────────────────────────────────────────────

def _model_config(settings: Dict[str, Any]):
    from hospital_client.model_management.config import ModelConfig
    from hospital_client.model_management.registry import get_architecture_info
    return ModelConfig(architecture=settings["architecture"], num_classes=len(settings["class_mapping"]),
                       input_size=get_architecture_info(settings["architecture"]).default_input_size, color_mode="RGB")


def evaluate_parameters(parameters, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Global model metrics on the Module 5 test manifest (classification loss, accuracy, F1, per class)."""
    import torch
    from torch.utils.data import DataLoader
    from hospital_client.model_management.registry import build_model
    from hospital_client.model_management.parameters import set_parameters
    from hospital_client.model_management.device import resolve_device
    from hospital_client.training.dataset import ManifestImageDataset, load_manifest
    from hospital_client.training.metrics import compute_classification_metrics

    cfg = _model_config(settings)
    device = resolve_device(settings["device"])
    model = build_model(cfg)
    set_parameters(model, parameters)
    model.to(device).eval()
    test = load_manifest(os.path.join(settings["dataset_dir"], "test_manifest.json"))
    loader = DataLoader(ManifestImageDataset(test, settings["class_mapping"], cfg.color_mode, tuple(cfg.input_size)),
                        batch_size=settings["batch_size"], shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, n, y_true, y_pred = 0.0, 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            total += loss_fn(out, labels).item() * labels.size(0)
            n += labels.size(0)
            y_true += labels.cpu().tolist()
            y_pred += out.argmax(1).cpu().tolist()
    return compute_classification_metrics(y_true, y_pred, total / n if n else 0.0, settings["class_mapping"]).to_dict()


# ── Client process ─────────────────────────────────────────────────────────────────────────

def run_client(k: int, address: str, settings: Dict[str, Any], client_dir: str, lock) -> None:
    import flwr as fl
    from hospital_client.model_management.registry import build_model
    from hospital_client.model_management.parameters import get_parameters, set_parameters
    from hospital_client.model_management.store import ModelStore
    from hospital_client.training.config import TrainingConfig
    from hospital_client.training.federation import _compute_parameters_checksum
    from hospital_client.training.trainer import Trainer

    has_val = os.path.exists(os.path.join(client_dir, "validation_manifest.json"))
    with open(os.path.join(client_dir, "train_manifest.json")) as f:
        n_train = len(json.load(f)["records"])  # FedAvg/FedProx weight of this client

    class HospitalClient(fl.client.NumPyClient):
        def fit(self, parameters, config):
            rnd = int(config["server_round"])
            mu = float(config.get("proximal_mu", 0.0))  # set by Flower's FedProx strategy
            cfg = _model_config(settings)
            work = os.path.join(settings["output_dir"], "work", f"client_{k}", f"round_{rnd}")
            shutil.rmtree(work, ignore_errors=True)
            store = ModelStore(os.path.join(work, "models"))  # = Module 7's model store for output_dir=work
            model = build_model(cfg)
            set_parameters(model, parameters)
            store.save_checkpoint(model, cfg, f"GLOBAL_R{rnd}", status="available", version=1)
            tc = TrainingConfig(
                model_id=f"CLIENT{k}_R{rnd}", model_config=cfg, dataset_dir=client_dir, output_dir=work,
                require_validation=has_val, device=settings["device"], epochs=settings["epochs"],
                batch_size=settings["batch_size"], learning_rate=settings["learning_rate"],
                random_seed=settings["seed"] + 1000 * rnd + k, canonical_class_mapping=settings["class_mapping"],
                canonical_base_model_id=f"GLOBAL_R{rnd}", canonical_base_model_version=1,
                canonical_base_model_checksum=_compute_parameters_checksum(get_parameters(model)),
                proximal_mu=mu,
            )
            with lock:  # one client trains at a time
                result = Trainer(tc).run()
            if result.checkpoint_model_id is None:
                raise RuntimeError(f"Client {k} produced no checkpoint in round {rnd}: {result.errors}")
            trained, _ = store.load_checkpoint(result.checkpoint_model_id, result.checkpoint_version)
            params = get_parameters(trained)
            shutil.rmtree(work, ignore_errors=True)  # checkpoints are not kept; metrics are returned
            metrics = {"client": k, "epochs_run": result.epochs_run, "seconds": round(result.training_duration_seconds, 2),
                       "train_loss": result.training_metrics.get("loss", 0.0),
                       "train_accuracy": result.training_metrics.get("accuracy", 0.0),
                       "val_accuracy": (result.validation_metrics or {}).get("accuracy", -1.0), "proximal_mu": mu}
            return params, n_train, metrics

        def evaluate(self, parameters, config):  # evaluation is server-side on the test split
            return 0.0, 1, {}

    time.sleep(3)  # let the server bind first
    fl.client.start_client(server_address=address, client=HospitalClient().to_client(), insecure=True)


# ── Server / orchestration ─────────────────────────────────────────────────────────────────

def _watch_clients(procs, done) -> None:
    """Flower's server waits up to 24 h for missing clients; a crashed client ends the run instead."""
    while not done.wait(2):
        for p in procs:
            if p.exitcode not in (None, 0):
                print(f"Error: client process {p.name} exited with code {p.exitcode}; stopping the simulation "
                      "(see the client's error above).", file=sys.stderr, flush=True)
                os._exit(1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_simulation(settings: Dict[str, Any]) -> Dict[str, Any]:
    import flwr as fl
    import torch
    from hospital_client.model_management.registry import build_model
    from hospital_client.model_management.parameters import get_parameters
    from hospital_client.training.dataset import derive_class_mapping, load_manifest

    if not os.path.exists(os.path.join(settings["dataset_dir"], "test_manifest.json")):
        raise ValueError("The dataset needs a test_manifest.json: the server evaluates every round on the test split.")
    os.makedirs(settings["output_dir"], exist_ok=True)
    settings["class_mapping"] = derive_class_mapping(load_manifest(os.path.join(settings["dataset_dir"], "train_manifest.json")))
    clients = write_client_datasets(settings["dataset_dir"], os.path.join(settings["output_dir"], "clients"),
                                    settings["clients"], settings["partition"], settings["alpha"], settings["seed"])

    torch.manual_seed(settings["seed"])
    initial = get_parameters(build_model(_model_config(settings)))
    model_bytes = int(sum(a.nbytes for a in initial))

    rounds: List[Dict[str, Any]] = []
    client_metrics: Dict[int, List[Dict[str, Any]]] = {}
    clock = {"t": time.time()}

    def evaluate_fn(server_round, parameters, config):
        m = evaluate_parameters(parameters, settings)
        now = time.time()
        rounds.append({"round": server_round, "test": m, "seconds": round(now - clock["t"], 2),
                       "bytes_exchanged": 0 if server_round == 0 else 2 * model_bytes * settings["clients"],
                       "clients": client_metrics.get(server_round, [])})
        clock["t"] = now
        print(f"[round {server_round}/{settings['rounds']}] test_accuracy={m['accuracy']:.4f} test_f1={m['f1']:.4f}", flush=True)
        return float(m["loss"]), {"accuracy": float(m["accuracy"]), "f1": float(m["f1"])}

    def fit_metrics(results):
        # Called right before evaluate_fn of the same round.
        client_metrics[len(rounds)] = sorted((dict(m, num_examples=n) for n, m in results), key=lambda m: m["client"])
        return {}

    common = dict(
        fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=settings["clients"],
        min_available_clients=settings["clients"], min_evaluate_clients=0,
        initial_parameters=fl.common.ndarrays_to_parameters(initial), evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: {"server_round": rnd}, fit_metrics_aggregation_fn=fit_metrics,
    )
    strategy = (fl.server.strategy.FedProx(proximal_mu=settings["mu"], **common) if settings["strategy"] == "fedprox"
                else fl.server.strategy.FedAvg(**common))

    address = f"127.0.0.1:{_free_port()}"
    ctx = mp.get_context("spawn")
    lock = ctx.Lock()
    procs = [ctx.Process(target=run_client, args=(c["client"], address, settings, c["dataset_dir"], lock), daemon=True)
             for c in clients]
    for p in procs:
        p.start()
    started = time.time()
    done = threading.Event()
    threading.Thread(target=_watch_clients, args=(procs, done), daemon=True).start()
    try:
        fl.server.start_server(server_address=address, config=fl.server.ServerConfig(num_rounds=settings["rounds"]), strategy=strategy)
    finally:
        done.set()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    shutil.rmtree(os.path.join(settings["output_dir"], "work"), ignore_errors=True)

    results = {
        "simulation_version": SIM_VERSION, "flower_version": fl.__version__,
        "configuration": {k: v for k, v in settings.items() if k != "class_mapping"},
        "class_mapping": settings["class_mapping"], "model_bytes": model_bytes,
        "partition": clients, "rounds": rounds, "total_seconds": round(time.time() - started, 2),
        "completed_rounds": max(0, len(rounds) - 1),
    }
    with open(os.path.join(settings["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(settings["output_dir"], "rounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "test_loss", "test_accuracy", "test_f1", "seconds", "bytes_exchanged"])
        for r in rounds:
            w.writerow([r["round"], r["test"]["loss"], r["test"]["accuracy"], r["test"]["f1"], r["seconds"], r["bytes_exchanged"]])
    if results["completed_rounds"] != settings["rounds"]:
        raise RuntimeError(f"Only {results['completed_rounds']} of {settings['rounds']} rounds completed; see the log above.")
    # Flower reports a client's training error to the server instead of crashing; such a round
    # aggregates fewer updates (or none), so the run is not a valid experiment.
    short = [r["round"] for r in rounds[1:] if len(r["clients"]) != settings["clients"]]
    if short:
        raise RuntimeError(f"Not every client returned an update in round(s) {short}; see the client errors above. "
                           f"Partial results were written to {settings['output_dir']}.")
    return results


def parse_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(prog="python -m federation.flower_sim", description="Flower federated simulation bench")
    p.add_argument("--dataset", dest="dataset_dir", required=True, help="Module 5 output dir (train/validation/test manifests)")
    p.add_argument("--output", dest="output_dir", required=True)
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--partition", choices=PARTITIONS, default="iid")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for label_skew (smaller = more skew)")
    p.add_argument("--strategy", choices=STRATEGIES, default="fedavg")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx proximal_mu (required > 0 with --strategy fedprox)")
    p.add_argument("--architecture", required=True)
    p.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=42)
    s = vars(p.parse_args(argv))
    if s["clients"] < 1 or s["rounds"] < 1 or s["epochs"] < 1:
        p.error("--clients, --rounds and --epochs must be >= 1")
    if s["strategy"] == "fedprox" and not s["mu"] > 0:
        p.error("--strategy fedprox needs --mu > 0")
    if s["strategy"] == "fedavg" and s["mu"]:
        p.error("--mu is only used with --strategy fedprox")
    s["dataset_dir"], s["output_dir"] = os.path.abspath(s["dataset_dir"]), os.path.abspath(s["output_dir"])
    return s


def main(argv: Optional[List[str]] = None) -> None:
    try:
        results = run_simulation(parse_args(argv))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    last = results["rounds"][-1]["test"]
    print(f"Done: {results['completed_rounds']} rounds, final test accuracy {last['accuracy']:.4f}. "
          f"Results in {os.path.join(results['configuration']['output_dir'], 'results.json')}")


if __name__ == "__main__":
    main()
