import argparse
import sys

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import build_model, list_architectures
from hospital_client.model_management.store import ModelStore


def _add_init_parser(subparsers):
    p = subparsers.add_parser("init", help="Construct a new untrained model and save it as version 1")
    p.add_argument("--architecture", required=True, help=f"One of: {list_architectures()}")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--output", required=True, help="Artifact store root directory")
    p.add_argument("--model-id", default=None, help="Defaults to the architecture name if omitted")
    p.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=[224, 224])
    p.add_argument("--color-mode", choices=["RGB", "L"], default="RGB")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=None)


def _add_inspect_parser(subparsers):
    p = subparsers.add_parser("inspect", help="Validate and print metadata for a stored model version")
    p.add_argument("model_id")
    p.add_argument("--store", required=True, help="Artifact store root directory")
    p.add_argument("--version", type=int, default=None, help="Defaults to the latest version if omitted")


def _add_list_parser(subparsers):
    p = subparsers.add_parser("list", help="List all versions of a model")
    p.add_argument("model_id")
    p.add_argument("--store", required=True, help="Artifact store root directory")


def main():
    parser = argparse.ArgumentParser(description="Module 6: Model Management")
    subparsers = parser.add_subparsers(dest="command")
    _add_init_parser(subparsers)
    _add_inspect_parser(subparsers)
    _add_list_parser(subparsers)

    args = parser.parse_args()

    try:
        if args.command == "init":
            config = ModelConfig(
                architecture=args.architecture,
                num_classes=args.num_classes,
                input_size=tuple(args.input_size),
                color_mode=args.color_mode,
                pretrained=args.pretrained,
                dropout=args.dropout,
            )
            model = build_model(config)
            store = ModelStore(args.output)
            model_id = args.model_id or args.architecture
            metadata = store.save_checkpoint(model, config, model_id, status="draft")
            print(f"Initialized '{model_id}' version {metadata.version} ({args.architecture}) in {args.output}")

        elif args.command == "inspect":
            store = ModelStore(args.store)
            version = args.version if args.version is not None else store.get_latest(args.model_id).version
            result = store.validate(args.model_id, version)
            print(f"Model: {args.model_id}  Version: {version}")
            print(f"Valid: {result.valid}")
            if result.metadata is not None:
                print(f"Architecture: {result.metadata.architecture}")
                print(f"Status: {result.metadata.status}")
                print(f"Parameters: {result.metadata.total_parameters}")
            if result.reasons:
                print(f"Reasons: {result.reasons}")
            if not result.valid:
                sys.exit(1)

        elif args.command == "list":
            store = ModelStore(args.store)
            versions = store.list_versions(args.model_id)
            if not versions:
                print(f"No versions found for model '{args.model_id}'.")
            for m in versions:
                print(f"v{m.version}  status={m.status}  architecture={m.architecture}  created_at={m.created_at}")

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
