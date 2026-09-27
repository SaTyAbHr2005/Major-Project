
import argparse
import sys
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest
from hospital_client.dataset.ingestion.report_generator import generate_reports

def main():
    parser = argparse.ArgumentParser(description="Module 4: Dataset Ingestion and Inspection")
    parser.add_argument("dataset_path", help="Path to the dataset (directory for images, file for CSV/Excel)")
    parser.add_argument("--output", default=".", help="Directory to save the generated reports")
    
    args = parser.parse_args()
    
    print(f"Starting inspection for: {args.dataset_path}")
    try:
        profile = detect_and_ingest(args.dataset_path)
        generate_reports(profile, args.output)
        print(f"Inspection complete. Reports generated in: {args.output}")
        if profile.errors:
            print(f"WARNING: Encountered {len(profile.errors)} errors during inspection.")
            sys.exit(0) # Not failing the build just because there are invalid files
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

