from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
OUTPUT_PATH = ROOT / "submissions" / "test_baseline_last_month.csv"

MONTH_COL = "\u041c\u0435\u0441\u044f\u0446"
TARGET_COL = "\u0420\u0422\u041e"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    last_month = train[MONTH_COL].max()
    last_month_rto = (
        train.loc[train[MONTH_COL] == last_month, ["new_id", TARGET_COL]]
        .rename(columns={TARGET_COL: "rto"})
        .copy()
    )

    fallback_rto = train[TARGET_COL].median()
    submission = test[["new_id"]].merge(last_month_rto, on="new_id", how="left")
    submission["rto"] = submission["rto"].fillna(fallback_rto).clip(lower=0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(submission)}")
    print(f"Last month used: {last_month}")
    print(f"Fallback rto: {fallback_rto:.2f}")


if __name__ == "__main__":
    main()
