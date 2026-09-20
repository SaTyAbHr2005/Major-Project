import argparse
import json
import sys

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.provisioning import LocalStoreModelProvider
from hospital_client.inference.result import format_summary


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-root", required=True,
                   help="Module 6 artifact store root (for a Module 7 run this is <training output>/models)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--version", type=int, required=True, help="Explicit model version - 'latest' is never assumed")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")


def _engine(args) -> LocalInferenceEngine:
    provider = LocalStoreModelProvider(args.models_root, args.model_id, args.version)
    return LocalInferenceEngine(provider, InferenceConfig(device=args.device))


def main():
    parser = argparse.ArgumentParser(description="Module 16: Local Inference (the image never leaves this machine)")
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="Run local inference on one image")
    _add_model_args(predict)
    predict.add_argument("--image", required=True, help="Local image path (read-only; never copied or uploaded)")
    predict.add_argument("--reference", default=None, help="Opaque caller-supplied identifier echoed in the result")
    predict.add_argument("--json", action="store_true", help="Print the full result as JSON")

    validate = subparsers.add_parser("validate-model", help="Load and validate a model without running inference")
    _add_model_args(validate)

    args = parser.parse_args()
    try:
        if args.command == "predict":
            engine = _engine(args)
            try:
                result = engine.predict(args.image, args.reference)
            finally:
                engine.close()
            print(json.dumps(result.to_dict(), indent=2) if args.json else format_summary(result))
            if not result.succeeded:
                sys.exit(1)
        elif args.command == "validate-model":
            engine = _engine(args)
            print(json.dumps(engine.model_info(), indent=2))
            engine.close()
        else:
            parser.print_help()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
