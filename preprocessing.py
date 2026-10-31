
import os

# Increase Hugging Face download timeouts.
# Set these before importing Hugging Face libraries.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

import io
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset, Audio


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_NAME = "mteb/urbansound8K"

TARGET_SR = 22050
DURATION = 4.0

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

EXPECTED_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

MAX_ROWS_TO_SCAN = 8732

OUTPUT_DIR = "outputs"
SPECTROGRAM_DIR = os.path.join(OUTPUT_DIR, "spectrograms")

os.makedirs(SPECTROGRAM_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET IN STREAMING MODE
# ============================================================

print("\nLoading UrbanSound8K in streaming mode...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

# IMPORTANT:
# Disable automatic audio decoding.
# This prevents Hugging Face from using TorchCodec.

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

print("Dataset loaded in streaming mode!")

print("\nDataset features:")
print(dataset.features)


# ============================================================
# 3. MANUAL AUDIO DECODING
# ============================================================

def decode_audio_manually(audio_info):
    """
    Decode audio using SoundFile instead of TorchCodec.

    audio_info contains:
        bytes: raw audio bytes
        path: audio file path

    Returns:
        audio_array: NumPy array
        sampling_rate: original sampling rate
    """

    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:

        audio_array, sampling_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True
        )

    elif audio_path is not None and os.path.exists(audio_path):

        audio_array, sampling_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

    else:
        raise ValueError(
            "Audio data contains neither valid bytes nor a valid path."
        )

    return audio_array, sampling_rate


# ============================================================
# 4. CONVERT STEREO TO MONO
# ============================================================

def convert_to_mono(audio):

    """
    SoundFile returns audio in shape:

        (number_of_samples, number_of_channels)

    For stereo:
        Left channel + Right channel
        --------------------------------
                       2

    """

    if audio.ndim == 2:

        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


# ============================================================
# 5. RESAMPLE AUDIO
# ============================================================

def resample_audio(audio, original_sr):

    """
    Convert all audio recordings to a common sampling rate:
    22050 Hz.

    Resampling changes the number of samples, not the
    actual duration of the recording.
    """

    if original_sr != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    return audio.astype(np.float32)


# ============================================================
# 6. PREPARE FIXED-LENGTH AUDIO FOR VISUALIZATION
# ============================================================

def fix_audio_length(audio):

    """
    Make the visualization input exactly 4 seconds long.

    If audio is longer:
        Truncate it.

    If audio is shorter:
        Add zero padding.

    The original processed waveform is retained separately.
    """

    required_samples = int(TARGET_SR * DURATION)

    if len(audio) > required_samples:

        audio = audio[:required_samples]

    elif len(audio) < required_samples:

        padding = required_samples - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    return audio.astype(np.float32)


# ============================================================
# 7. GENERATE WAVEFORM, STFT AND MEL SPECTROGRAM
# ============================================================

def generate_visualizations(audio, class_name, filename):

    """
    Generate three visualizations:

    1. Waveform
    2. STFT Spectrogram
    3. Mel Spectrogram

    """

    # Fixed 4-second input for consistent visualization.
    plot_audio = fix_audio_length(audio)

    # --------------------------------------------------------
    # A. WAVEFORM
    # --------------------------------------------------------

    time_axis = np.arange(len(plot_audio)) / TARGET_SR

    # --------------------------------------------------------
    # B. STFT SPECTROGRAM
    # --------------------------------------------------------

    stft = librosa.stft(
        plot_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    magnitude = np.abs(stft)

    stft_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    # --------------------------------------------------------
    # C. MEL SPECTROGRAM
    # --------------------------------------------------------

    mel_spec = librosa.feature.melspectrogram(
        y=plot_audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # --------------------------------------------------------
    # PLOT ALL THREE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 12)
    )

    fig.suptitle(
        f"Audio Preprocessing - {class_name}",
        fontsize=16
    )

    # Waveform
    axes[0].plot(time_axis, plot_audio)

    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    # STFT Spectrogram
    img1 = librosa.display.specshow(
        stft_db,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=axes[1]
    )

    axes[1].set_title("STFT Spectrogram")

    fig.colorbar(
        img1,
        ax=axes[1],
        format="%+2.0f dB"
    )

    # Mel Spectrogram
    img2 = librosa.display.specshow(
        mel_spec_db,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[2]
    )

    axes[2].set_title("Mel Spectrogram")

    fig.colorbar(
        img2,
        ax=axes[2],
        format="%+2.0f dB"
    )

    plt.tight_layout()

    output_path = os.path.join(
        SPECTROGRAM_DIR,
        f"{class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved visualization: {output_path}")


# ============================================================
# 8. FIND ONE AUDIO RECORDING FROM EACH CLASS
# ============================================================

print("\nSearching for one recording from each class...\n")

selected_classes = {}

metadata = []

for row in tqdm(dataset, total=MAX_ROWS_TO_SCAN):

    class_name = row["class"]

    # Skip classes for which we already have an example.
    if class_name in selected_classes:
        continue

    print(f"\nProcessing class: {class_name}")

    try:

        # Decode raw audio bytes using SoundFile.
        audio_array, original_sr = decode_audio_manually(
            row["audio"]
        )

        # Convert stereo to mono.
        audio_array = convert_to_mono(audio_array)

        # Resample to 22050 Hz.
        processed_audio = resample_audio(
            audio_array,
            original_sr
        )

        # Generate waveform and spectrogram visualizations.
        generate_visualizations(
            processed_audio,
            class_name,
            row["slice_file_name"]
        )

        # Store the original processed waveform.
        selected_classes[class_name] = processed_audio

        # Preserve official dataset metadata.
        metadata.append({

            "filename": row["slice_file_name"],

            "class": class_name,

            "classID": row["classID"],

            "fold": row["fold"],

            "original_sampling_rate": original_sr,

            "processed_sampling_rate": TARGET_SR,

            "original_duration": len(audio_array) / original_sr,

            "processed_duration": len(processed_audio) / TARGET_SR,

            "number_of_samples": len(processed_audio)

        })

        print(
            f"Successfully processed {class_name}"
        )

    except Exception as error:

        print(
            f"Error processing {class_name}: {error}"
        )

        continue

    # Stop once we have one example of every class.
    if len(selected_classes) == len(EXPECTED_CLASSES):

        print("\nFound examples from all 10 classes!")

        break


# ============================================================
# 9. SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "preprocessing_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("PREPROCESSING COMPLETED")

print("=" * 60)

print(
    f"\nNumber of classes processed: "
    f"{len(selected_classes)}/{len(EXPECTED_CLASSES)}"
)

print("\nClasses found:")

for class_name in selected_classes:

    print(f"  - {class_name}")

missing_classes = set(EXPECTED_CLASSES) - set(selected_classes)

if missing_classes:

    print("\nClasses not processed:")

    for class_name in sorted(missing_classes):

        print(f"  - {class_name}")

print("\nMetadata:")

print(metadata_df)

print("\nOutput directory:")

print(os.path.abspath(OUTPUT_DIR))

print("\nPreprocessing pipeline finished!")
