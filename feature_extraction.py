
import os

# Set download timeouts before importing Hugging Face libraries
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

import io
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from tqdm import tqdm
from datasets import load_dataset, Audio


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_NAME = "mteb/urbansound8K"

TARGET_SR = 22050

N_FFT = 2048
HOP_LENGTH = 512

N_MFCC = 13

EXPECTED_ROWS = 8732

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET IN STREAMING MODE
# ============================================================

print("\nLoading UrbanSound8K dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

# Disable automatic TorchCodec decoding
dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

print("Dataset loaded successfully!")


# ============================================================
# 3. MANUALLY DECODE AUDIO
# ============================================================

def decode_audio_manually(audio_info):

    """
    Decode audio using SoundFile.

    This avoids TorchCodec and PyTorch DLL dependencies.
    """

    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:

        audio, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True
        )

    elif audio_path is not None and os.path.exists(audio_path):

        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

    else:

        raise ValueError("Audio bytes/path not available")

    return audio, sr


# ============================================================
# 4. AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio, original_sr):

    """
    Convert stereo to mono and resample to 22050 Hz.
    """

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample if required
    if original_sr != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    return audio.astype(np.float32)


# ============================================================
# 5. HELPER: CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

def calculate_statistics(feature_matrix):

    """
    Calculate mean and standard deviation across time frames.

    Input:
        Feature matrix of shape (number_of_features, number_of_frames)

    Output:
        Mean and standard deviation for every feature.
    """

    mean_values = np.mean(
        feature_matrix,
        axis=1
    )

    std_values = np.std(
        feature_matrix,
        axis=1
    )

    return mean_values, std_values


# ============================================================
# 6. EXTRACT BASELINE FEATURES (26 DIMENSIONS)
# ============================================================

def extract_baseline_features(audio):

    """
    Extract:
        13 MFCC means
        13 MFCC standard deviations

    Total = 26 features
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean, mfcc_std = calculate_statistics(mfcc)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features


# ============================================================
# 7. EXTRACT EXTENDED FEATURES (58 DIMENSIONS)
# ============================================================

def extract_extended_features(audio):

    """
    Extract:

        MFCC:
            13 means + 13 standard deviations = 26

        Delta MFCC:
            13 means + 13 standard deviations = 26

        Spectral centroid:
            Mean + standard deviation = 2

        Spectral rolloff:
            Mean + standard deviation = 2

        Zero crossing rate:
            Mean + standard deviation = 2

    Total = 58 features
    """

    # --------------------------------------------------------
    # A. MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean, mfcc_std = calculate_statistics(mfcc)

    # --------------------------------------------------------
    # B. DELTA MFCC
    # --------------------------------------------------------

    number_of_frames = mfcc.shape[1]

    if number_of_frames >= 3:

        # Delta width must be odd and no larger than
        # the number of available frames.
        delta_width = min(9, number_of_frames)

        if delta_width % 2 == 0:
            delta_width -= 1

        delta_mfcc = librosa.feature.delta(
            mfcc,
            width=delta_width,
            order=1,
            axis=-1,
            mode="nearest"
        )

    else:

        # Extremely short clips: use zero delta features
        delta_mfcc = np.zeros_like(mfcc)

    delta_mean, delta_std = calculate_statistics(
        delta_mfcc
    )

    # --------------------------------------------------------
    # C. SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    centroid_mean, centroid_std = calculate_statistics(
        centroid
    )

    # --------------------------------------------------------
    # D. SPECTRAL ROLLOFF
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85
    )

    rolloff_mean, rolloff_std = calculate_statistics(
        rolloff
    )

    # --------------------------------------------------------
    # E. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    zcr_mean, zcr_std = calculate_statistics(
        zcr
    )

    # --------------------------------------------------------
    # F. COMBINE ALL 58 FEATURES
    # --------------------------------------------------------

    features = np.concatenate([

        # 26 MFCC features
        mfcc_mean,
        mfcc_std,

        # 26 Delta MFCC features
        delta_mean,
        delta_std,

        # 2 Spectral centroid features
        centroid_mean,
        centroid_std,

        # 2 Spectral rolloff features
        rolloff_mean,
        rolloff_std,

        # 2 Zero crossing rate features
        zcr_mean,
        zcr_std

    ])

    return features


# ============================================================
# 8. FEATURE COLUMN NAMES
# ============================================================

baseline_columns = (

    [f"mfcc_{i+1}_mean" for i in range(N_MFCC)]

    +

    [f"mfcc_{i+1}_std" for i in range(N_MFCC)]

)


extended_columns = (

    baseline_columns

    +

    [f"delta_mfcc_{i+1}_mean" for i in range(N_MFCC)]

    +

    [f"delta_mfcc_{i+1}_std" for i in range(N_MFCC)]

    +

    ["spectral_centroid_mean", "spectral_centroid_std"]

    +

    ["spectral_rolloff_mean", "spectral_rolloff_std"]

    +

    ["zcr_mean", "zcr_std"]

)


# ============================================================
# 9. PROCESS THE COMPLETE DATASET
# ============================================================

print("\nStarting feature extraction...")
print("This will process all 8732 recordings.\n")

baseline_rows = []
extended_rows = []

processed_count = 0
failed_count = 0

for row in tqdm(dataset, total=EXPECTED_ROWS):

    try:

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------

        filename = row["slice_file_name"]

        class_name = row["class"]

        class_id = row["classID"]

        fold = row["fold"]

        # ----------------------------------------------------
        # Decode audio
        # ----------------------------------------------------

        audio, original_sr = decode_audio_manually(
            row["audio"]
        )

        # ----------------------------------------------------
        # Preprocess audio
        # ----------------------------------------------------

        audio = preprocess_audio(
            audio,
            original_sr
        )

        # ----------------------------------------------------
        # Extract baseline features
        # ----------------------------------------------------

        baseline_features = extract_baseline_features(
            audio
        )

        # ----------------------------------------------------
        # Extract extended features
        # ----------------------------------------------------

        extended_features = extract_extended_features(
            audio
        )

        # ----------------------------------------------------
        # Validate feature dimensions
        # ----------------------------------------------------

        if len(baseline_features) != 26:

            raise ValueError(
                f"Expected 26 baseline features, "
                f"got {len(baseline_features)}"
            )

        if len(extended_features) != 58:

            raise ValueError(
                f"Expected 58 extended features, "
                f"got {len(extended_features)}"
            )

        # ----------------------------------------------------
        # Check for invalid numerical values
        # ----------------------------------------------------

        if not np.all(np.isfinite(baseline_features)):

            raise ValueError("Invalid baseline feature values")

        if not np.all(np.isfinite(extended_features)):

            raise ValueError("Invalid extended feature values")

        # ----------------------------------------------------
        # Common metadata
        # ----------------------------------------------------

        metadata = {

            "filename": filename,

            "class": class_name,

            "classID": class_id,

            "fold": fold,

            "original_sampling_rate": original_sr,

            "processed_sampling_rate": TARGET_SR,

            "duration": len(audio) / TARGET_SR

        }

        # ----------------------------------------------------
        # Store baseline row
        # ----------------------------------------------------

        baseline_row = metadata.copy()

        for i, column in enumerate(baseline_columns):

            baseline_row[column] = baseline_features[i]

        baseline_rows.append(baseline_row)

        # ----------------------------------------------------
        # Store extended row
        # ----------------------------------------------------

        extended_row = metadata.copy()

        for i, column in enumerate(extended_columns):

            extended_row[column] = extended_features[i]

        extended_rows.append(extended_row)

        processed_count += 1

    except Exception as error:

        failed_count += 1

        print(
            f"\nFailed: {row.get('slice_file_name', 'Unknown')} "
            f"| Error: {error}"
        )

        continue


# ============================================================
# 10. CREATE DATAFRAMES
# ============================================================

print("\nCreating feature DataFrames...")

baseline_df = pd.DataFrame(baseline_rows)

extended_df = pd.DataFrame(extended_rows)


# ============================================================
# 11. SAVE CSV FILES
# ============================================================

baseline_path = os.path.join(
    OUTPUT_DIR,
    "baseline_features.csv"
)

extended_path = os.path.join(
    OUTPUT_DIR,
    "extended_features.csv"
)

baseline_df.to_csv(
    baseline_path,
    index=False
)

extended_df.to_csv(
    extended_path,
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("FEATURE EXTRACTION COMPLETED")

print("=" * 65)

print(f"\nSuccessfully processed: {processed_count}")

print(f"Failed recordings: {failed_count}")

print("\nBaseline feature dataset:")

print("Shape:", baseline_df.shape)

print("Feature count:", len(baseline_columns))

print("\nExtended feature dataset:")

print("Shape:", extended_df.shape)

print("Feature count:", len(extended_columns))

print("\nClass distribution:")

if not extended_df.empty:

    print(extended_df["class"].value_counts())

print("\nOfficial fold distribution:")

if not extended_df.empty:

    print(extended_df["fold"].value_counts().sort_index())

print("\nSaved files:")

print(os.path.abspath(baseline_path))

print(os.path.abspath(extended_path))

print("\nFeature extraction pipeline finished!")
