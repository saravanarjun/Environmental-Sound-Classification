
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

baseline = pd.read_csv(OUTPUT_DIR / "baseline_features.csv")
extended = pd.read_csv(OUTPUT_DIR / "extended_features.csv")

metadata_cols = [
    "filename",
    "class",
    "classID",
    "fold",
    "original_sampling_rate",
    "processed_sampling_rate",
    "duration"
]

def inspect_dataset(name, df, expected_features):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols]

    print("Shape:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Missing values:", X.isna().sum().sum())
    print("Infinite values:", np.isinf(X.to_numpy()).sum())
    print("Duplicate filenames:", df["filename"].duplicated().sum())

    # Features that have the same value for every recording
    constant_features = X.columns[X.nunique() <= 1].tolist()
    print("Constant features:", constant_features)

    print("\nFeature statistics:")
    print(X.describe().loc[["mean", "std", "min", "max"]].round(3))

    assert len(feature_cols) == expected_features
    assert df.shape[0] == 8732
    assert df["filename"].is_unique

    return feature_cols


baseline_cols = inspect_dataset("BASELINE", baseline, 26)
extended_cols = inspect_dataset("EXTENDED", extended, 58)

print("\n" + "=" * 55)
print("BASELINE vs EXTENDED ALIGNMENT")
print("=" * 55)

same_recordings = (
    baseline[metadata_cols].equals(extended[metadata_cols])
)

print("Same recordings in same order:", same_recordings)
print("Same class distribution:",
      baseline["class"].value_counts().sort_index().equals(
          extended["class"].value_counts().sort_index()
      ))
print("Same fold distribution:",
      baseline["fold"].value_counts().sort_index().equals(
          extended["fold"].value_counts().sort_index()
      ))

if not same_recordings:
    print("\nWARNING: Baseline and extended metadata do not match!")

print("\nValidation completed.")
