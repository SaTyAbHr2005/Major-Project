import argparse
import json
import sys

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.result import format_summary
from hospital_client.training.trainer import Trainer


def _add_common_training_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", dest="dataset_dir", required=True, help="Module 5 output directory containing train/validation/test manifests")
    p.add_argument("--model", dest="architecture", required=True, help="Module 6 architecture identifier (e.g. efficientnet_b0)")
    p.add_argument("--model-id", default=None, help="Local model_id for the Module 6 artifact store; defaults to --model")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=[224, 224])
    p.add_argument("--color-mode", choices=["RGB", "L"], default="RGB")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--output", dest="output_dir", required=True, help="Directory for the Module 6 artifact store and Module 7 training state")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["fp32", "fp16", "auto"], default="fp32", help="fp32 (default, safe baseline) | fp16 (CUDA-only) | auto (fp16 on CUDA, fp32 on CPU)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--scheduler", choices=["none", "reduce_on_plateau", "step", "cosine"], default="none")
    p.add_argument("--scheduler-patience", type=int, default=2, help="reduce_on_plateau")
    p.add_argument("--scheduler-factor", type=float, default=0.5, help="reduce_on_plateau")
    p.add_argument("--scheduler-step-size", type=int, default=10, help="step")
    p.add_argument("--scheduler-gamma", type=float, default=0.1, help="step")
    p.add_argument("--scheduler-t-max", type=int, default=None, help="cosine; defaults to --epochs when unset")
    p.add_argument("--scheduler-eta-min", type=float, default=0.0, help="cosine")
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--early-stopping-metric", choices=["val_loss", "val_accuracy", "val_f1"], default="val_loss")
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--class-weighting", choices=["none", "manifest", "balanced"], default="none")
    p.add_argument("--no-validation", action="store_true", help="Train without requiring a validation manifest")
    p.add_argument("--test", dest="run_test_evaluation", action="store_true", help="Run final evaluation on the test manifest")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", dest="random_seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")


def _config_from_args(args) -> TrainingConfig:
    model_config = ModelConfig(
        architecture=args.architecture,
        num_classes=args.num_classes,
        input_size=tuple(args.input_size),
        color_mode=args.color_mode,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    return TrainingConfig(
        model_id=args.model_id or args.architecture,
        model_config=model_config,
        dataset_dir=args.dataset_dir,
        require_validation=not args.no_validation,
        run_test_evaluation=args.run_test_evaluation,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_eta_min=args.scheduler_eta_min,
        early_stopping=args.early_stopping,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_patience=args.early_stopping_patience,
        class_weighting=args.class_weighting,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        resume=args.resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Module 7: Local Deep Learning Training")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run local training")
    _add_common_training_args(train_parser)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a training configuration without training")
    _add_common_training_args(validate_parser)

    args = parser.parse_args()

    try:
        if args.command == "train":
            config = _config_from_args(args)
            result = Trainer(config).run()
            print(format_summary(result, args.architecture))
            result_path = f"{args.output_dir}/{config.model_id}_training_result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nFull result written to: {result_path}")
            if result.status.value == "TRAINING_FAILED":
                sys.exit(1)

        elif args.command == "validate-config":
            config = _config_from_args(args)
            validate_training_config(config)
            print("Training configuration is valid.")

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
