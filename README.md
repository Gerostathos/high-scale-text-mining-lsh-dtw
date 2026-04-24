# High-Scale Text Mining, LSH, and Dynamic Time Warping

This project implements a high-scale analytics pipeline for three data mining tasks:

1. **News article text classification** using Bag-of-Words features with SVM, Random Forest, k-NN, and an optional transformer-based DeBERTa experiment.
2. **Approximate nearest-neighbor search** using Jaccard similarity, MinHash, and Locality Sensitive Hashing (LSH).
3. **Dynamic Time Warping (DTW)** implemented from scratch for time-series similarity.

The work was developed for a Big Data Mining / High-Scale Analytics assignment and focuses on scalable preprocessing, model evaluation, nearest-neighbor retrieval, and algorithmic implementation.

---

## Project Motivation

Large text datasets create practical challenges in preprocessing, feature extraction, classification, and nearest-neighbor search. This project explores classical and scalable data mining techniques, including sparse Bag-of-Words representations, linear SVMs, Random Forests, brute-force k-NN, MinHash LSH, and DTW for temporal sequence comparison.

The goal is not only to train models, but also to compare accuracy, runtime, scalability, and implementation trade-offs.

---

## Main Features

- Loads and explores a large news classification dataset.
- Combines article titles and content into a unified text field.
- Applies text preprocessing such as lowercasing, punctuation removal, stopword removal, stemming, lemmatization, number removal, and whitespace normalization.
- Uses Bag-of-Words features for classification.
- Trains and evaluates SVM and Random Forest models.
- Uses 5-fold cross-validation for evaluation.
- Applies Optuna for hyperparameter tuning.
- Includes optional GPU-oriented experiments using spaCy, Hugging Face, cuML, and DeBERTa.
- Implements brute-force k-NN with Jaccard similarity.
- Implements MinHash LSH and LSH Forest variants for approximate nearest-neighbor search.
- Compares LSH build time, query time, total time, overlap with exact neighbors, and accuracy.
- Implements Dynamic Time Warping from scratch for time-series distance computation.

---

## Repository Structure

```text
high-scale-text-mining-lsh-dtw/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── high_scale_analytics_experiments.ipynb
│
├── data/
│   └── README.md
│
└── reports/
    └── README.md
```

The datasets and generated model/output files are not committed to GitHub. The `data/README.md` file explains where the datasets should be placed locally.

---

## Dataset

The text classification and LSH experiments use the Kaggle Big Data 2024 classification dataset. The assignment dataset contains news articles with the following fields:

- `Id`
- `Title`
- `Content`
- `Label` for the training set only

The four target categories are:

- `Business`
- `Entertainment`
- `Health`
- `Technology`

The DTW task uses a separate time-series dataset containing pairs of sequences.

Expected local structure:

```text
data/
├── bigdata2024classification/
│   ├── train.csv
│   └── test_without_labels.csv
│
├── pickle-files/
│   ├── processed_data.pkl
│   └── processed_test_data.pkl
│
└── timeseriesexercise3/
    └── dtw_test.csv
```

File names may differ depending on whether the code is run locally or on Kaggle. The notebook contains Kaggle-style paths for reproducibility in the original execution environment.

---

## Methods

### 1. Text Classification

The classification pipeline combines `Title` and `Content`, preprocesses the text, vectorizes it using Bag-of-Words, and evaluates classical machine-learning models.

Implemented models include:

- Support Vector Machine (SVM)
- Random Forest
- k-NN with Jaccard similarity
- optional DeBERTa transformer fine-tuning experiment

### 2. Hyperparameter Optimization

Optuna is used to tune model and vectorizer settings, including:

- n-gram range,
- maximum number of features,
- minimum and maximum document frequency,
- SVM regularization,
- SVM kernel,
- Random Forest tree depth,
- number of estimators,
- split and leaf constraints,
- class weighting.

### 3. Nearest-Neighbor Search and LSH

The project compares exact brute-force k-NN against approximate nearest-neighbor methods based on MinHash LSH.

Evaluated variants include:

- brute-force Jaccard k-NN,
- standard MinHash LSH,
- LSH Forest,
- vectorized MinHash LSH.

The comparison includes runtime, exact-neighbor overlap, and classification accuracy.

### 4. Dynamic Time Warping

DTW is implemented manually rather than using a prebuilt DTW library. The implementation builds a dynamic-programming matrix and computes the minimum alignment cost between two numeric sequences.

---

## Representative Results

### Text Classification

| Model | Evaluation | Result |
|---|---:|---:|
| SVM with Bag-of-Words | 5-fold CV mean accuracy | 0.9669 |
| SVM with Bag-of-Words | 5-fold CV standard deviation | 0.0013 |
| Random Forest with Bag-of-Words | 5-fold CV mean accuracy | 0.6887 |
| Random Forest with Bag-of-Words | 5-fold CV standard deviation | 0.0057 |

The SVM performed best because linear margin-based classifiers are well suited to high-dimensional sparse text representations.

### k-NN and LSH

| Method | Build Time (s) | Query Time (s) | Total Time (s) | Fraction Overlap | Accuracy |
|---|---:|---:|---:|---:|---:|
| Brute-Force Jaccard | 0.00 | 155.32 | 155.32 | 1.0000 | 0.9650 |
| LSH-Jaccard-16 | 104.88 | 766.49 | 871.38 | 0.2075 | 0.8349 |
| LSH-Jaccard-32 | 108.53 | 77.75 | 186.27 | 0.2846 | 0.8843 |
| LSH-Jaccard-64 | 115.12 | 127.35 | 242.47 | 0.4062 | 0.9321 |
| LSH-Forest-16 | 100.21 | 25.58 | 125.79 | 0.0477 | 0.6236 |
| LSH-Forest-32 | 107.14 | 28.51 | 135.64 | 0.1055 | 0.7257 |
| LSH-Forest-64 | 115.02 | 33.76 | 148.78 | 0.1006 | 0.7205 |
| Vectorized LSH-16 | 83.47 | 1119.03 | 1202.50 | 0.2088 | 0.8346 |
| Vectorized LSH-32 | 93.75 | 105.13 | 198.88 | 0.2871 | 0.8805 |
| Vectorized LSH-64 | 102.50 | 205.15 | 307.65 | 0.4114 | 0.9298 |

### Dynamic Time Warping

The DTW implementation was used to process 1002 sequence pairs. The reported runtime was approximately 22 minutes on Kaggle CPU.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Gerostathos/high-scale-text-mining-lsh-dtw.git
cd high-scale-text-mining-lsh-dtw
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some GPU-specific libraries used in the notebook, such as RAPIDS cuML, are not installed through this requirements file because their installation depends on CUDA, Python version, and operating system. The notebook can still be reviewed without installing those optional GPU packages.

### 4. Add the datasets locally

Place the datasets under:

```text
data/
```

See `data/README.md` for the expected folder layout.

### 5. Run the notebook

```bash
jupyter notebook notebooks/high_scale_analytics_experiments.ipynb
```

If running locally instead of Kaggle, update the Kaggle-style input paths in the notebook to point to your local `data/` folder.

---

## Important Implementation Notes

The notebook was originally executed in a Kaggle / local experimental environment, so some paths may need to be adapted when running on another machine.

Recommended path pattern for a cleaner local version:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
TEXT_DATA_DIR = DATA_DIR / "bigdata2024classification"
DTW_DATA_DIR = DATA_DIR / "timeseriesexercise3"

TRAIN_PATH = TEXT_DATA_DIR / "train.csv"
TEST_PATH = TEXT_DATA_DIR / "test_without_labels.csv"
DTW_PATH = DTW_DATA_DIR / "dtw_test.csv"
```

---

## What This Project Demonstrates

This repository demonstrates:

- large-scale text preprocessing,
- sparse text feature engineering,
- Bag-of-Words classification,
- SVM and Random Forest model comparison,
- Optuna hyperparameter tuning,
- k-NN with Jaccard similarity,
- approximate nearest-neighbor search with MinHash LSH,
- runtime and accuracy trade-off analysis,
- manual Dynamic Time Warping implementation,
- large-notebook experimentation and reporting.

---

## Future Improvements

- Refactor the notebook into reusable modules under `src/`.
- Add command-line scripts for classification, LSH evaluation, and DTW computation.
- Replace Kaggle-style paths with a configuration file.
- Save final metrics to CSV files automatically.
- Add small sample datasets for quick reproducible demos.
- Add figures for preprocessing effects, class distribution, and runtime comparisons.
