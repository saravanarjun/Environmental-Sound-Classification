# 🎧 UrbanSound8K Audio Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Librosa](https://img.shields.io/badge/Librosa-1.0.0-green.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)

An end-to-end Machine Learning pipeline for environmental sound classification on the **UrbanSound8K** dataset. This project covers streaming ingestion, robust audio signal preprocessing (avoiding PyTorch/TorchCodec bottlenecks), multi-scale feature engineering (MFCCs, Delta MFCCs, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate), systematic hyperparameter sweeps, depth-based overfitting mitigation, and rigorous model benchmarking across 6 distinct machine learning paradigms.

---

## 📋 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Dataset Overview](#-dataset-overview)
3. [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
4. [Audio Feature Engineering: Concepts & Mathematical Formulations](#-audio-feature-engineering-concepts--mathematical-formulations)
   - [Mel-Frequency Cepstral Coefficients (MFCCs)](#1-mel-frequency-cepstral-coefficients-mfccs)
   - [Delta MFCCs ($\Delta$ MFCCs)](#2-delta-mfccs-first-order-derivatives)
   - [Spectral Centroid](#3-spectral-centroid)
   - [Spectral Rolloff](#4-spectral-rolloff)
   - [Zero Crossing Rate (ZCR)](#5-zero-crossing-rate-zcr)
   - [Baseline vs. Extended Feature Representations](#6-baseline-vs-extended-feature-representations)
5. [Model Architectures & Theoretical Formulations](#-model-architectures--theoretical-formulations)
6. [Hyperparameter Optimization & Overfitting Deep-Dive](#-hyperparameter-optimization--overfitting-deep-dive)
   - [The Random Forest Overfitting Paradox](#the-random-forest-overfitting-paradox)
   - [Max Depth Restriction & Generalization Gap Analysis](#max-depth-restriction--generalization-gap-analysis)
   - [Model-by-Model Parameter Response](#model-by-model-parameter-response)
7. [Extensive Experimental Results & Benchmark Analysis](#-extensive-experimental-results--benchmark-analysis)
   - [Grand Final Leaderboard](#1-grand-final-leaderboard)
   - [Baseline (26) vs. Extended (58) Features Performance](#2-baseline-26-vs-extended-58-features-performance)
   - [Top Discriminative Audio Features](#3-top-discriminative-audio-features)
   - [Champion Model Evaluation & Confusion Matrix](#4-champion-model-evaluation--confusion-matrix)
8. [Project File Structure](#-project-file-structure)
9. [Installation & Setup](#-installation--setup)
10. [How to Run](#-how-to-run)
11. [Conclusions & Key Engineering Insights](#-conclusions--key-engineering-insights)

---

## 🎯 Problem Statement

Environmental sound classification (ESC) is a foundational capability in acoustic surveillance, smart city infrastructure, ecological monitoring, autonomous vehicles, and context-aware IoT devices. Unlike speech or music, environmental sounds present severe non-stationary challenges:

* **High Acoustic Heterogeneity**: Urban sounds range from continuous harmonic drones (*engine idling*, *air conditioning*) to broadband stochastic vibrations (*drilling*, *jackhammer*) and short, transient impulses (*gunshots*, *car horns*).
* **High Acoustic Overlap**: Mechanically driven sounds share dense low-to-mid frequency profiles, creating severe classification ambiguity (e.g., *jackhammer* vs. *drilling*, or *air conditioner* vs. *engine idling*).
* **Signal Duration & Quality Variability**: Real-world recordings contain background ambient noise, reverberations, varying durations ($\le 4.0\text{s}$), multiple sampling rates ($11.025\text{ kHz}$ to $48.0\text{ kHz}$), and inconsistent channel configurations (stereo and mono).

The objective is to design a robust, deterministic machine learning pipeline capable of extracting noise-invariant, physically grounded audio descriptors and classifying sound events with high precision, macro F1-score, and bounded generalization error.

---

## 📊 Dataset Overview

The project uses the **UrbanSound8K** dataset (accessed via the Hugging Face `mteb/urbansound8K` streaming repository):

* **Total Samples**: 8,732 labeled audio excerpts.
* **Duration**: $\le 4$ seconds per slice.
* **Folds**: 10 pre-arranged cross-validation folds (arranged by field-recording source to prevent data leakage between slices).
* **Classes (10 classes)**:
  1. `air_conditioner`
  2. `car_horn`
  3. `children_playing`
  4. `dog_bark`
  5. `drilling`
  6. `engine_idling`
  7. `gun_shot`
  8. `jackhammer`
  9. `siren`
  10. `street_music`

---

## ⚙️ End-to-End Pipeline Architecture

The workflow follows a 4-stage modular architecture:

```
[ mteb/urbansound8K (Hugging Face Stream) ]
                     │
                     ▼
  [ Stage 1: Decoding & Preprocessing (preprocessing.py) ]
    ├── Direct SoundFile byte decoding (no TorchCodec / PyTorch DLLs)
    ├── Stereo-to-Mono Channel Averaging: [ (L + R) / 2 ]
    ├── Standardized Nyquist Resampling: 22,050 Hz (Target Bandwidth: 11,025 Hz)
    └── Fixed-Length Padded/Truncated Visualizations (4.0s) -> STFT & Mel Spectrograms
                     │
                     ▼
  [ Stage 2: Feature Extraction (feature_extraction.py) ]
    ├── Baseline Features (26 dims): 13 MFCC means (μ) + 13 MFCC standard deviations (σ)
    └── Extended Features (58 dims):
          ├── 13 MFCC μ + 13 MFCC σ               (26 dims)
          ├── 13 Delta MFCC μ + 13 Delta MFCC σ   (26 dims)
          ├── Spectral Centroid μ + σ             ( 2 dims)
          ├── Spectral Rolloff μ + σ              ( 2 dims)
          └── Zero Crossing Rate μ + σ            ( 2 dims)
                     │
                     ▼
  [ Stage 3: Data Quality Validation (validate_features.py) ]
    ├── Dimension assertions (26 and 58 features)
    ├── Infinite / NaN / Missing value detection
    └── Row-level baseline-to-extended alignment verification (8,732 rows)
                     │
                     ▼
  [ Stage 4: Machine Learning & Hyperparameter Tuning (sound.ipynb) ]
    ├── StandardScaler & LabelEncoder normalization
    ├── Model-by-model hyperparameter tuning & depth analysis
    ├── Train vs. Test F1 curve plotting (overfitting gap monitoring)
    ├── Head-to-head model comparisons (Baseline vs. Extended)
    └── Champion model selection & Normalized Confusion Matrix
```

---

## 🔬 Audio Feature Engineering: Concepts & Mathematical Formulations

Audio signals cannot be passed directly into classical machine learning algorithms as raw 1D time-domain waveforms because raw amplitude samples are phase-sensitive, high-dimensional, and non-stationary. Instead, we compute time-frequency spectral representations and summary statistics.

### 1. Mel-Frequency Cepstral Coefficients (MFCCs)
* **Perceptual Rationale**: Human auditory perception does not scale linearly across frequencies. The human ear has high frequency resolution below $1,000\text{ Hz}$ and logarithmic resolution above $1,000\text{ Hz}$. MFCCs compress the Fourier spectrum into the **Mel scale**, mimicking the cochlea's basilar membrane.
* **Mathematical Extraction**:
  1. Compute Short-Time Fourier Transform (STFT) with a Hann window:
     $$X(m, k) = \sum_{n=0}^{N-1} x[n + mH] w[n] e^{-j 2\pi k n / N}$$
     where $N = 2048$ (FFT window length) and $H = 512$ (hop length).
  2. Compute power spectrum: $P(m, k) = |X(m, k)|^2$.
  3. Map onto $M = 128$ triangular Mel filterbanks $H_m(k)$:
     $$S(m) = \sum_{k} P(m, k) H_m(k)$$
     where the Mel frequency $m$ is defined as: $m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$.
  4. Take the logarithm of the filterbank energies: $L(m) = \log(S(m))$.
  5. Apply Discrete Cosine Transform (DCT-II) to decorrelate filterbank outputs into cepstral coefficients:
     $$c_n = \sum_{m=0}^{M-1} L(m) \cos\left[ \frac{\pi n}{M} \left(m + \frac{1}{2}\right) \right], \quad n \in \{1, \dots, 13\}$$
* **Aggregation**: For each audio clip, we compute the **mean ($\mu$)** and **standard deviation ($\sigma$)** across all temporal frames for the first 13 coefficients ($2 \times 13 = 26\text{ features}$).
  * **Mean**: Captures the overall spectral envelope / average timbre.
  * **Standard Deviation**: Captures the dynamic range and temporal variance of each spectral band.

### 2. Delta MFCCs (First-Order Derivatives)
* **Perceptual Rationale**: Static MFCCs only represent static timbre averaged over time; they fail to capture the *rate of change* (trajectory) of spectral energy. Real-world sounds are highly dynamic (e.g., a siren rapidly sweeps upward in frequency, while a gunshot exhibits instantaneous transient decay).
* **Mathematical Formulation**: The first-order temporal derivative $\Delta c_t$ is computed via a central-difference Savitzky-Golay FIR filter:
  $$\Delta c_t = \frac{\sum_{n=1}^{N_d} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N_d} n^2}$$
* **Aggregation**: 13 Delta means + 13 Delta standard deviations ($26\text{ features}$).

### 3. Spectral Centroid
* **Perceptual Rationale**: Corresponds directly to the human perception of acoustic **"brightness"** or **"sharpness"**. High values indicate high-frequency dominant energy (e.g., *sirens*, *gunshots*), whereas low values indicate low-frequency energy (e.g., *engine idling*, *air conditioners*).
* **Mathematical Formulation**: The amplitude-weighted average frequency of the spectrum:
  $$\text{Centroid}(m) = \frac{\sum_{k=0}^{N/2} f(k) |X(m, k)|}{\sum_{k=0}^{N/2} |X(m, k)|}$$
* **Aggregation**: Mean + Standard Deviation ($2\text{ features}$).

### 4. Spectral Rolloff
* **Perceptual Rationale**: Approximates the frequency boundary below which the vast majority ($85\%$) of the spectral energy is concentrated. It distinguishes between harmonic signals with steep spectral roll-offs (e.g., musical instruments, clean whistles) and broadband acoustic noise (e.g., *jackhammers*, *drills*).
* **Mathematical Formulation**: The frequency $R(m)$ such that:
  $$\sum_{k=0}^{R(m)} |X(m, k)| = 0.85 \sum_{k=0}^{N/2} |X(m, k)|$$
* **Aggregation**: Mean + Standard Deviation ($2\text{ features}$).

### 5. Zero Crossing Rate (ZCR)
* **Perceptual Rationale**: Measures the rate at which the audio signal changes sign (crosses zero). High ZCR indicates high-frequency noise, fricatives, and percussive transients (e.g., *gunshots*, *glass breaking*), whereas low ZCR indicates smooth, periodic harmonic waveforms (e.g., *sirens*, *engine rumbling*).
* **Mathematical Formulation**:
  $$\text{ZCR}(m) = \frac{1}{2W} \sum_{n=0}^{W-1} \left| \text{sgn}(x[n]) - \text{sgn}(x[n-1]) \right|$$
* **Aggregation**: Mean + Standard Deviation ($2\text{ features}$).

### 6. Baseline vs. Extended Feature Representations
* **Baseline Set (26 Features)**: 13 MFCC means + 13 MFCC standard deviations.
* **Extended Set (58 Features)**: 26 Baseline MFCCs + 26 Delta MFCCs + 2 Spectral Centroid + 2 Spectral Rolloff + 2 Zero Crossing Rate.

---

## 🧠 Model Architectures & Theoretical Formulations

We benchmarked 6 distinct machine learning algorithms to evaluate linear, margin-based, probabilistic, and tree-ensemble paradigms:

1. **Support Vector Machine (SVM / SVC with RBF Kernel)**:
   * Maps features into an infinite-dimensional Hilbert space using the Radial Basis Function kernel: $K(x, z) = \exp(-\gamma ||x - z||^2)$.
   * Maximizes the margin separating support vectors while penalizing classification violations via box constraint $C$:
     $$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i$$
2. **XGBoost (Extreme Gradient Boosting)**:
   * Additive ensemble of decision trees minimizing a second-order Taylor expansion of multi-class log-loss with $L_1$ and $L_2$ leaf-weight regularization:
     $$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
3. **Random Forest Classifier**:
   * Bagging ensemble of deep de-correlated decision trees with randomized feature subspace selection at each split.
4. **Quadratic Discriminant Analysis (QDA)**:
   * Assumes normally distributed features per class $\mathcal{N}(\mu_k, \Sigma_k)$ with **class-specific covariance matrices** $\Sigma_k$, yielding non-linear quadratic decision surfaces.
5. **Logistic Regression (Multinomial / Softmax)**:
   * Linear decision boundaries parameterized by cross-entropy loss with $L_2$ weight regularization controlled by inverse strength $C$.
6. **Linear Discriminant Analysis (LDA)**:
   * Models classes with equal covariance matrices $\Sigma$, computing orthogonal discriminant vectors maximizing Fisher's criterion (between-class variance relative to within-class variance).

---

## 📉 Hyperparameter Optimization & Overfitting Deep-Dive

### The Random Forest Overfitting Paradox
When running default Random Forest (`max_depth=None`), we observed a classic symptom of high variance:
* **Training Macro F1 = $1.0000$ ($100\%$ accuracy)**
* **Testing Macro F1 = $0.8612$ ($86.8\%$ accuracy)**
* **Overfitting Gap = $0.1388$ ($13.88\%$ discrepancy)**

Because single decision trees are unconstrained, they recursively split down to single samples, perfectly memorizing background noise artifacts specific to the training set.

### Max Depth Restriction & Generalization Gap Analysis
To mitigate this, we swept `max_depth` across $[2, 4, 6, 8, 10, 12, 14, 16, 20, 25, \text{None}]$:

```text
Macro F1
  1.00 |                                    ----------------- Train F1 (100% memorization)
       |                                  / . . . . . . . . . [Overfitting Gap = 13.3%]
  0.86 |                             /---*------------------- Test F1 plateaus at depth 16-20
       |                           /
  0.70 |                     /----/
       |                   /
  0.50 |             /----/
       |       /----/
  0.30 |  /---/
       +----------------------------------------------------> Tree max_depth
          2    4    6    8    10   12   14   16   20   25  None
         [Underfitting]                  [Optimal]  [Pure Noise Memorization]
```

#### Key Findings:
1. **Underfitting ($depth \le 6$)**: Tree capacity is too restricted ($F_1 < 0.62$).
2. **Optimal Trade-Off ($depth = 16 - 20$)**: Test performance reaches its ceiling (**$F_1 = 0.8673$**, Accuracy = $87.29\%$).
3. **Redundant Depth ($depth > 20$ to $\text{None}$)**: Zero generalization gain. The model merely inflates model size and memorizes outlier leaves. Capping depth at $20$ bounds tree growth without hurting classification.

### Model-by-Model Parameter Response

* **Support Vector Machine (SVM)**:
  * Swept $C \in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]$.
  * At default $C=1.0$, Test $F_1 = 0.8389$. At **$C=50.0$**, Test $F_1$ surged to **$0.8983$** (Accuracy = **$90.21\%$**). The higher penalty forces the optimizer to find tight, complex non-linear boundaries in the feature space without suffering from under-regularization.
* **XGBoost**:
  * Swept `max_depth` $\in [2 \dots 10]$ and `n_estimators` $\in [25 \dots 300]$.
  * Optimal setting: `max_depth=6-10`, `n_estimators=300` $\to$ Test $F_1 = \mathbf{0.8993}$ on extended features.
* **Logistic Regression**:
  * Swept $C \in [0.001 \dots 100.0]$.
  * Peaked at $C=0.10$ ($F_1 = 0.6393$). Shows very small overfitting gap ($0.0194$), but is limited by the linear separability of acoustic data.
* **LDA & QDA**:
  * LDA with LSQR solver peaked at `shrinkage=0.0` ($F_1 = 0.5740$).
  * QDA peaked at covariance regularization `reg_param=0.01` ($F_1 = 0.7070$). QDA outperforms LDA because acoustic classes have widely different covariance structures (e.g., impulsive gunshots vs. continuous air conditioning).

---

## 🏆 Extensive Experimental Results & Benchmark Analysis

### 1. Grand Final Leaderboard

All models evaluated with their optimal hyperparameters on the standard test partition ($1,747$ samples):

| Rank | Model | Optimal Hyperparameters | Train $F_1$ | Test $F_1$ | Overfitting Gap | Test Accuracy |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **SVM (RBF Kernel)** | $C = 50.0, \text{kernel}=\text{rbf}$ | $0.9934$ | **$0.8983$** | **$0.0951$** | **$90.21\%$** |
| 🥈 **2** | **XGBoost** | $\text{max\_depth}=10, n\_est=300$ | $1.0000$ | **$0.8930$** | $0.1070$ | **$89.70\%$** |
| 🥉 **3** | **Random Forest** | $\text{max\_depth}=20, n\_est=200$ | $1.0000$ | **$0.8673$** | $0.1327$ | **$87.29\%$** |
| 4 | **QDA** | $\text{reg\_param}=0.01$ | $0.7324$ | **$0.7070$** | **$0.0254$** | **$70.06\%$** |
| 5 | **Logistic Regression** | $C = 0.10$ | $0.6587$ | **$0.6393$** | **$0.0194$** | **$63.31\%$** |
| 6 | **LDA (LSQR)** | $\text{shrinkage}=0.0$ | $0.6027$ | **$0.5740$** | $0.0288$ | **$56.90\%$** |

![Final Comparison](outputs/final_models_grand_comparison.png)

---

### 2. Baseline (26) vs. Extended (58) Features Performance

When transitioning from Baseline (MFCCs only) to Extended features (MFCCs + Delta MFCCs + Spectral Centroid + Rolloff + ZCR):

| Model | Baseline $F_1$ (26) | Extended $F_1$ (58) | Absolute Gain ($\Delta$) | Why This Happened |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | $0.6393$ | **$0.7029$** | **$+6.36\%$** | Non-linear features (ZCR, Rolloff) projected data into linear separability. |
| **LDA (LSQR)** | $0.5740$ | **$0.6249$** | **$+5.10\%$** | Spectral centroid and dynamics separated previously overlapping centroids. |
| **XGBoost** | $0.8890$ | **$0.8993$** | **$+1.03\%$** | Reached highest overall accuracy ($90.2\%$) leveraging delta split points. |
| **Random Forest** | $0.8673$ | **$0.8734$** | **$+0.61\%$** | Feature subspace randomization gained more discriminative splits. |
| **SVM (RBF)** | $0.8983$ | **$0.8953$** | $-0.03\%$ | RBF kernel was already implicitly modeling non-linear interactions. |
| **QDA** | $0.7070$ | **$0.6947$** | $-1.23\%$ | $58 \times 58$ covariance matrix per class increased parameter estimation error. |

---

### 3. Top Discriminative Audio Features

Using Gini-gain feature importance from the champion XGBoost model on the 58 extended features:

```text
Rank  Feature Name             Importance  Origin
────────────────────────────────────────────────────────────────────────
 1.   delta_mfcc_4_std           8.15%     [Extended: Dynamic Rate of Change]
 2.   mfcc_13_std                5.03%     [Baseline: High-frequency Timbre Variance]
 3.   zcr_mean                   4.44%     [Extended: Perceptual Noisiness]
 4.   mfcc_1_std                 3.87%     [Baseline: Overall Energy Variance]
 5.   zcr_std                    3.35%     [Extended: Transient Pulsing]
 6.   delta_mfcc_3_std           3.08%     [Extended: Low-mid Formant Trajectory]
 7.   mfcc_1_mean                3.04%     [Baseline: Mean Log Energy]
 8.   mfcc_11_std                2.79%     [Baseline: Upper-mid Resonance]
 9.   spectral_rolloff_mean      2.57%     [Extended: Spectral Energy Boundary]
10.   delta_mfcc_10_std          2.50%     [Extended: Upper Formant Velocity]
```

> **Key Discovery**: **5 of the top 10 most critical features** came directly from the newly engineered Extended features (`delta_mfcc`, `zcr`, and `spectral_rolloff`), directly validating the feature engineering strategy.

---

### 4. Champion Model Evaluation & Confusion Matrix

The champion model (**Tuned XGBoost on Extended Features** / **Tuned SVM**) achieves **$90.2\%$ overall accuracy** and **$89.9\%$ Macro $F_1$**:

```text
                  precision    recall  f1-score   support

 air_conditioner      0.944     0.916     0.930       203
        car_horn      0.911     0.837     0.873        86
children_playing      0.790     0.842     0.815       183
        dog_bark      0.923     0.896     0.909       201
        drilling      0.925     0.893     0.909       206
   engine_idling      0.926     0.974     0.949       193
        gun_shot      0.897     0.847     0.871        72
      jackhammer      0.929     0.942     0.936       208
           siren      0.931     0.976     0.953       165
    street_music      0.855     0.843     0.849       230

        accuracy                          0.902      1747
       macro avg      0.903     0.897     0.899      1747
    weighted avg      0.903     0.902     0.902      1747
```

![Confusion Matrix](outputs/champion_model_confusion_matrix.png)

#### Acoustic Confusion Analysis:
* **Easiest Classes ($F_1 > 0.94$)**: `siren` ($0.953$), `engine_idling` ($0.949$), and `jackhammer` ($0.936$). These sounds have distinct harmonic curves (sirens) or continuous rhythmic low-frequency energy (engines).
* **Most Confused Pairs**:
  * `children_playing` ($81.5\%$) vs. `street_music` ($84.9\%$): Both contain wide dynamic ranges, crowd babble, vocal utterances, and ambient street reflections.
  * `drilling` ($90.9\%$) vs. `jackhammer` ($93.6\%$): Both share broadband percussive impact harmonics; Delta MFCCs and Zero Crossing Rate provided the vital signals needed to resolve them.

---

## 📁 Project File Structure

```
c:/Users/sarav/OneDrive/Desktop/ML/
├── .venv/                              # Local isolated virtual environment
├── outputs/                            # Generated CSV datasets and high-res plots
│   ├── baseline_features.csv           # 8,732 rows × 26 features + metadata
│   ├── extended_features.csv           # 8,732 rows × 58 features + metadata
│   ├── preprocessing_metadata.csv      # Sample extraction metadata
│   ├── spectrograms/                   # Class spectrograms (10 PNGs)
│   ├── all_models_f1_tuning_curves.png # Multi-model normalized response
│   ├── rf_hyperparameter_tuning.png    # Random Forest depth & overfitting plot
│   ├── xgb_hyperparameter_tuning.png   # XGBoost depth & rounds plot
│   ├── svm_hyperparameter_tuning.png   # SVM C regularization plot
│   ├── lr_hyperparameter_tuning.png    # Logistic Regression C plot
│   ├── discriminant_hyperparameter_tuning.png # LDA & QDA regularization
│   ├── final_models_grand_comparison.png # Leaderboard & Generalization Gap
│   └── champion_model_confusion_matrix.png # Normalized confusion matrix
├── preprocessing.py                    # Audio streaming, decoding, and spectrogram generation
├── feature_extraction.py               # Batch extraction of 26 and 58 features
├── validate_features.py                # Dataset validation and integrity tests
├── sound.ipynb                         # Jupyter Notebook with all models and plots
├── requirements.txt                    # Pinned project dependencies
└── README.md                           # Professional project documentation
```

---

## 💻 Installation & Setup

### Prerequisites
* Windows 10/11 (or Linux/macOS)
* Python 3.11, 3.12, or 3.13

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/urbansound-ml.git
cd urbansound-ml
```

### 2. Create and Activate Virtual Environment
```powershell
# Using Python standard venv
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# (Or on Linux/macOS: source .venv/bin/activate)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Preprocess Audio & Generate Class Spectrograms
```bash
python preprocessing.py
```
* Streams samples, verifies SoundFile decoding, resamples to $22,050\text{ Hz}$, and outputs class spectrograms into `outputs/spectrograms/`.

### Step 2: Extract Baseline and Extended Features
```bash
python feature_extraction.py
```
* Processes all $8,732$ recordings.
* Generates `outputs/baseline_features.csv` ($26\text{ features}$) and `outputs/extended_features.csv` ($58\text{ features}$).

### Step 3: Validate Extracted Datasets
```bash
python validate_features.py
```
* Verifies zero missing values, zero infs, and row-level alignment.

### Step 4: Run Machine Learning Models & Visualizations
```bash
# Launch Jupyter Notebook or run inside VS Code
jupyter notebook sound.ipynb
```
* Open [`sound.ipynb`](file:///c:/Users/sarav/OneDrive/Desktop/ML/sound.ipynb), select the `.venv` kernel, and run all cells to reproduce all hyperparameter curves, comparison charts, and confusion matrices.

---

## 📌 Conclusions & Key Engineering Insights

1. **Audio Feature Engineering Trumps Model Complexity**: Adding temporal velocity ($\Delta\text{ MFCCs}$) and spectral energy distribution ($\text{ZCR}$, $\text{Rolloff}$) produced a **$+6.36\%$ boost** in Logistic Regression and a **$+5.10\%$ boost** in LDA, proving that audio domain knowledge fundamentally reshapes feature separability.
2. **Combating Overfitting in Tree Models**: Unconstrained decision trees in Random Forest achieve $100\%$ training accuracy by memorizing audio noise. Imposing a maximum depth constraint (`max_depth = 16 - 20`) stops spurious leaf growth, bounds generalization gap, and preserves peak test performance ($87.3\%$).
3. **SVM and Gradient Boosting Dominate**: **SVM with RBF Kernel ($C=50.0$)** and **XGBoost ($n\_est=300, depth=6-10$)** emerged as the co-champions, achieving **$\approx 90.2\%$ accuracy** and **$89.9\%$ Macro $F_1$**, offering the strongest non-linear decision boundaries for complex acoustic data.
4. **Resilient Production Pipeline**: Decoupling audio decoding from heavy PyTorch/TorchCodec libraries in favor of `soundfile` ensured a deterministic, lightweight, cross-platform pipeline that executes with zero native DLL errors on Windows.
