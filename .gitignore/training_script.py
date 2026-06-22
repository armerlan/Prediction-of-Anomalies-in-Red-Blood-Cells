import os
import pickle
import pandas as pd
from pathlib import Path

from tabpfn import TabPFNClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tqdm import tqdm

from backend.config import settings

# allow large dataset usage on CPU
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# -----------------------------
# Paths
# -----------------------------

DATASET_PATH = settings.RECON_OUTPUT_FOLDER / "training_dataset.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "tabpfn_classifier.pkl"
FEATURE_PATH = MODEL_DIR / "features.pkl"

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(DATASET_PATH)

# convert labels
df["label"] = df["label"].map({"n": 0, "d": 1})

features = [
    "area_um2",
    "perimeter",
    "circularity",
    "elongation",
    "std_phase",
    "min_phase",
    "max_phase",
    "MCH_surface_density",
    "optical_volume"
]

X = df[features].values
y = df["label"].values

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# Cross Validation (with tqdm)
# -----------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

print("\nRunning Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train), total=5)):

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    clf = TabPFNClassifier(ignore_pretraining_limits=True)

    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)

    cv_scores.append(acc)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

print("\nCross Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", sum(cv_scores) / len(cv_scores))

# -----------------------------
# Train Final Model
# -----------------------------

print("\nTraining final model on full training data...")

clf = TabPFNClassifier(ignore_pretraining_limits=True)

clf.fit(X_train, y_train)

# -----------------------------
# Test Evaluation
# -----------------------------

print("\nEvaluating on test set...")

y_pred = clf.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------

with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

with open(FEATURE_PATH, "wb") as f:
    pickle.dump(features, f)

print("\nModel saved to:", MODEL_PATH)
print("Feature list saved to:", FEATURE_PATH)