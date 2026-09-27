import argparse
import sys
import os
import subprocess
from .train import train_pipeline
from .predict import run_test_inference

def main():
    parser = argparse.ArgumentParser(description="ML Challenge 2026: Business Entity Resolution CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train LightGBM matching model and optimize threshold")
    train_parser.add_argument("--sample-size", type=int, default=25000, help="Number of S1 training records to use")
    train_parser.add_argument("--val-size", type=int, default=5000, help="Number of S1 validation records")
    train_parser.add_argument("--artifact-path", type=str, default=None, help="Where to save model artifact")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run test inference and produce submission TSVs")
    predict_parser.add_argument("--artifact-path", type=str, default=None, help="Path to trained model artifact")
    predict_parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output TSVs")
    predict_parser.add_argument("--threshold", type=float, default=0.70, help="Decision probability threshold")
    predict_parser.add_argument("--max-test", type=int, default=None, help="Optional limit on test S1 records for dry runs")

    # validate command
    val_parser = subparsers.add_parser("validate", help="Run official submission validator")
    val_parser.add_argument("--matching", type=str, default="output/matching_results.tsv")
    val_parser.add_argument("--candidate", type=str, default="output/candidate_pairs.tsv")
    val_parser.add_argument("--test-dir", type=str, default="student_resource/dataset/test")

    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(
            sample_size=args.sample_size,
            val_size=args.val_size,
            artifact_path=args.artifact_path
        )
    elif args.command == "predict":
        run_test_inference(
            artifact_path=args.artifact_path,
            output_dir=args.output_dir,
            threshold=args.threshold,
            max_test_s1=args.max_test
        )
    elif args.command == "validate":
        validator_script = os.path.join("student_resource", "utils", "validate_submission.py")
        cmd = [
            sys.executable,
            validator_script,
            "--matching", args.matching,
            "--candidate", args.candidate,
            "--test-dir", args.test_dir
        ]
        print(f"Running validator: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
