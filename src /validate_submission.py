import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
DEFAULT_SUBMISSION_PATH = ROOT / "submissions" / "test_baseline_last_month.csv"
MAX_FILE_SIZE_BYTES = 1_000_000


def validate_submission(path: Path) -> None:
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(path)

    expected_columns = ["new_id", "rto"]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {submission.columns.tolist()}")

    if len(submission) != 20_615:
        raise ValueError(f"Expected 20615 rows, got {len(submission)}")

    if submission["new_id"].duplicated().any():
        raise ValueError("Submission contains duplicated new_id values")

    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values")

    if (submission["rto"] < 0).any():
        raise ValueError("Submission contains negative rto values")

    expected_ids = set(test["new_id"])
    actual_ids = set(submission["new_id"])
    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        extra = len(actual_ids - expected_ids)
        raise ValueError(f"new_id mismatch: missing={missing}, extra={extra}")

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File is too large: {file_size} bytes")

    print("Submission is valid")
    print(f"Path: {path}")
    print(f"Rows: {len(submission)}")
    print(f"File size: {file_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_SUBMISSION_PATH),
        help="Path to submission CSV",
    )
    args = parser.parse_args()
    validate_submission(Path(args.path))


if __name__ == "__main__":
    main()
