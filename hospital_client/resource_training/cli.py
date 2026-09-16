import argparse
import json
import sys

from hospital_client.resource_training.dataset_characteristics import inspect_dataset
from hospital_client.resource_training.estimator import HistoricalThroughputStore
from hospital_client.resource_training.policy import ResourcePolicy, default_policy
from hospital_client.resource_training.recommender import generate_recommendations
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
    train_parser.add_argument("--policy", dest="policy_path", default=None)

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
            policy = _load_policy(args.policy_path)
            profile = build_resource_profile(storage_path=args.output_dir, policy=policy)
            dataset = inspect_dataset(args.dataset_dir)
            history = HistoricalThroughputStore(_history_path(args.output_dir))
            rec_set = generate_recommendations(profile, policy, dataset, history)
            chosen = next((r for r in rec_set.recommendations if r.recommendation_type == args.choice), None)
            if chosen is None:
                available = [r.recommendation_type for r in rec_set.recommendations]
                print(f"Error: recommendation type '{args.choice}' was not generated for this hardware. Available: {available}")
                sys.exit(1)

            training_config = to_training_config(
                chosen, model_id=args.model_id, dataset_dir=args.dataset_dir,
                output_dir=args.output_dir, num_classes=dataset.num_classes,
            )
            result, stats, handoff = run_resource_aware_training(training_config, policy, _history_path(args.output_dir), profile)

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
