import joblib
from backend.segmentation import run_cellpose, extract_features_per_cell, load_cellpose_model, create_segmentation_overlay
import cv2
import numpy as np
import time
from tabpfn import TabPFNClassifier

print("Loading .pkl files and model...")
scaler = joblib.load("D:/Projects/Thesis/sem10/pipeline/models/scaler.pkl")
clf = joblib.load("D:/Projects/Thesis/sem10/pipeline/models/xgboost_trained.pkl")
features = joblib.load("D:/Projects/Thesis/sem10/pipeline/models/features.pkl")
threshold = joblib.load("D:/Projects/Thesis/sem10/pipeline/models/threshold.pkl")
cellpose_model = load_cellpose_model()

print("Reading image...")
phase_image = cv2.imread(str("D:/Projects/Thesis/sem10/pipeline/recon_output/a2_2_d__r6_2_d/phase.png"), cv2.IMREAD_GRAYSCALE)
if phase_image is None:
    print("Image not loaded properly")
    exit()

img_for_seg = phase_image.copy()

print("Standardizing image...")
img_for_seg = (img_for_seg - np.min(img_for_seg)) / (np.max(img_for_seg) - np.min(img_for_seg) + 1e-8)
img_for_seg = (img_for_seg * 255).astype(np.uint8)

img_for_seg = cv2.GaussianBlur(img_for_seg, (5,5), 0)

print("Running cellpose...")
masks = run_cellpose(img_for_seg, cellpose_model)
print("Unique mask values:", np.unique(masks))

print("Extracting features...")
df = extract_features_per_cell(
        phase_real=phase_image,
        masks=masks,
        pix_size=0.1,
        lambda_nm=633,
        alpha_cm3_per_g=0.18
    )

if df is None or len(df) == 0:
    print("No features extracted")

df = df.fillna(0)

missing = [f for f in features if f not in df.columns]
if missing:
    print("Missing features:", missing)
    exit()

X = df[features].values
print("Transforming X_scaled...")
X_scaled = scaler.transform(X)
print("Any NaNs in X_scaled:", np.isnan(X_scaled).any())
print("X_scaled shape:", X_scaled.shape)

# from tabpfn import TabPFNClassifier

# clf = TabPFNClassifier()
# clf.fit(X_scaled[:10], [0]*10)  # dummy fit just to initialize

probs = clf.predict_proba(X_scaled)
print("Works?")

cell_probs = clf.predict_proba(X_scaled)[:, 1]

df["cell_prob"] = cell_probs
df["cell_pred"] = (cell_probs > 0.5).astype(int)

image_prob = cell_probs.mean()
image_pred = int(image_prob > threshold)

label_map = {0: "Normal", 1: "Diabetic"}
prediction = label_map[image_pred]
print(prediction)