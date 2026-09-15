import argparse
import sys
from hospital_client.preprocessing.config import PreprocessingConfig, DatasetConfig, OutputMode
from hospital_client.preprocessing.engine import PreprocessingEngine

def main():
    parser = argparse.ArgumentParser(description="Module 5: Automated Preprocessing Engine")
    subparsers = parser.add_subparsers(dest="command")
    
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("dataset_path", help="Path to raw dataset")
    preprocess_parser.add_argument("--profile", required=True, help="Path to Module 4 profile JSON")
    preprocess_parser.add_argument("--output", required=True, help="Output directory")
    preprocess_parser.add_argument("--mode", choices=["lazy", "materialized"], default="lazy", help="Output mode")
    preprocess_parser.add_argument("--target-column", default=None, help="Explicit tabular target/label column (required if Module 4 only reported candidate columns)")
    preprocess_parser.add_argument("--group-id-map", default=None, help="Path to a JSON {rel_path: group_id} file with real patient/group identifiers, required for GROUPED strategy on image datasets")
    preprocess_parser.add_argument("--invalid-split-policy", choices=["error", "regenerate"], default="error", help="What to do when an existing split fails validation (INVALID/AMBIGUOUS): stop (default) or regenerate")

    args = parser.parse_args()

    if args.command == "preprocess":
        config = PreprocessingConfig()
        config.dataset.input_path = args.dataset_path
        config.dataset.output_path = args.output
        config.dataset.output_mode = OutputMode.LAZY if args.mode == "lazy" else OutputMode.MATERIALIZED
        config.tabular.target_column = args.target_column
        config.split.group_id_map_path = args.group_id_map
        config.split.invalid_split_policy = args.invalid_split_policy
        
        try:
            engine = PreprocessingEngine(config)
            engine.run(args.profile)
            print(f"Preprocessing completed. Check {args.output} for reports.")
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()

