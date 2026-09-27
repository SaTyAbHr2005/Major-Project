import argparse
import json
import sys

from hospital_client.resource_training.dataset_characteristics import inspect_dataset
from hospital_client.resource_training.estimator import HistoricalThroughputStore
from hospital_client.resource_training.policy import ResourcePolicy, default_policy
from hospital_client.resource_training.recommender import generate_recommendations, manual_config
from hospital_client.resource_training.resolved_config import to_training_config
from hospital_client.resource_training.resource_profile import build_resource_profile
from hospital_client.resource_training.runner import run_resource_aware_training


def _load_policy(path) -> ResourcePolicy:
    if not path:
        return default_policy()
    with open(path) as f:
        return ResourcePolicy.from_dict(json.load(f))


def _history_path(output_dir: str) -> str:
    return f"{output_dir}/resource_training_history.json"


def main():
    parser = argparse.ArgumentParser(description="Module 8: Resource-Aware Training")
    subparsers = parser.add_subparsers(dest="command")

    detect_parser = subparsers.add_parser("detect", help="Detect and print the local machine's ResourceProfile")
    detect_parser.add_argument("--output", dest="output_dir", default=".")

    recommend_parser = subparsers.add_parser("recommend", help="Generate resource-aware training recommendations")
    recommend_parser.add_argument("--dataset", dest="dataset_dir", required=True)
    recommend_parser.add_argument("--output", dest="output_dir", default=".")
    recommend_parser.add_argument("--policy", dest="policy_path", default=None)

    train_parser = subparsers.add_parser("train", help="Resolve a recommendation and run Module 7 training")
    train_parser.add_argument("--dataset", dest="dataset_dir", required=True)
    train_parser.add_argument("--output", dest="output_dir", required=True)
    train_parser.add_argument("--model-id", dest="model_id", required=True)
    train_parser.add_argument("--choice", choices=["recommended", "high_capacity", "fast"], default="recommended")
    train_parser.add_argument("--architecture", default=None,
                              help="Train this Module 6 architecture instead of a --choice (any model Module 8 assessed as feasible)")
    train_parser.add_argument("--batch-size", type=int, default=None,
                              help="With --architecture: batch size (at most the largest Module 8 measured as safe)")
    train_parser.add_argument("--epochs", type=int, default=None, help="With --architecture: number of epochs")
    train_parser.add_argument("--policy", dest="policy_path", default=None)
    train_parser.add_argument("--test", dest="run_test_evaluation", action="store_true", help="Run final evaluation on the test manifest")
    train_parser.add_argument("--plots", action="store_true", help="Write confusion matrix / ROC / PR plots for the test set (requires --test)")

    args = parser.parse_args()

    try:
        if args.command == "detect":
            profile = build_resource_profile(storage_path=args.output_dir)
            print(json.dumps(profile.to_dict(), indent=2))

        elif args.command == "recommend":
            # Pipeline order: M8 resource evaluation -> dataset evaluation -> model selection.
            policy = _load_policy(args.policy_path)
            profile = build_resource_profile(storage_path=args.output_dir, policy=policy)
            dataset = inspect_dataset(args.dataset_dir)
            history = HistoricalThroughputStore(_history_path(args.output_dir))
            rec_set = generate_recommendations(profile, policy, dataset, history)
            print(json.dumps(rec_set.to_dict(), indent=2))

        elif args.command == "train":
            # Full pipeline: M8 resource evaluation -> dataset evaluation ->
            # model selection -> training configuration -> M7 -> actual
            # training -> resource statistics -> M9 (see runner.py).
            if args.plots and not args.run_test_evaluation:
                raise ValueError("--plots requires --test (plots are generated from the test set).")
            if (args.batch_size is not None or args.epochs is not None) and not args.architecture:
                raise ValueError("--batch-size/--epochs require --architecture.")
            print("Re-checking hardware and regenerating recommendations (real local dry runs)...", flush=True)
            policy = _load_policy(args.policy_path)
            profile = build_resource_profile(storage_path=args.output_dir, policy=policy)
            dataset = inspect_dataset(args.dataset_dir)
            history = HistoricalThroughputStore(_history_path(args.output_dir))
            rec_set = generate_recommendations(profile, policy, dataset, history)
            if args.architecture:
                chosen = manual_config(rec_set, args.architecture, policy, args.batch_size, args.epochs)
            else:
                chosen = next((r for r in rec_set.recommendations if r.recommendation_type == args.choice), None)
            if chosen is None:
                available = [r.recommendation_type for r in rec_set.recommendations]
                print(f"Error: recommendation type '{args.choice}' was not generated for this hardware. Available: {available}")
                sys.exit(1)

            training_config = to_training_config(
                chosen, model_id=args.model_id, dataset_dir=args.dataset_dir,
                output_dir=args.output_dir, num_classes=dataset.num_classes,
                run_test_evaluation=args.run_test_evaluation,
            )
            print(f"Starting Module 7 training: {chosen.label} - {chosen.display_name}, {chosen.device}/{chosen.precision}, "
                  f"batch_size={chosen.batch_size}, epochs={chosen.epochs} (estimated {chosen.estimated_training_time_display})",
                  flush=True)
            result, stats, handoff = run_resource_aware_training(
                training_config, policy, _history_path(args.output_dir), profile, plots=args.plots)

            print(f"Chosen recommendation: {chosen.label} ({chosen.architecture})")
            print(f"Status: {result.status.value}")
            print(f"Actual training duration: {result.training_duration_seconds / 60.0:.2f} minutes "
                  f"(estimated range was: {chosen.estimated_training_time_display})")

            result_path = f"{args.output_dir}/{args.model_id}_training_result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            stats_path = f"{args.output_dir}/{args.model_id}_resource_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(stats.to_dict(), f, indent=2)
            print(f"Training result written to: {result_path}")
            print(f"Resource statistics written to: {stats_path}")
            if args.plots:
                print(f"Evaluation plots written to: {args.output_dir}/{args.model_id}_plots"
                      if result.test_metrics is not None else "No plots written: no test evaluation was produced.")

            if handoff is not None:
                handoff_dir = f"{args.output_dir}/{args.model_id}_federation_handoff"
                handoff.save(handoff_dir)
                print(f"Module 7 -> Module 9 FederationHandoff written to: {handoff_dir}")

            if result.status.value == "TRAINING_FAILED":
                sys.exit(1)

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
