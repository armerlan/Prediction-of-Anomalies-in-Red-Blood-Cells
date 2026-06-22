from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.config import settings
from backend.segmentation import (
    load_cellpose_model,
    run_cellpose,
    extract_features_per_cell
)

# -----------------------------
# Physical constants
# -----------------------------

PIX_SIZE = 0.1        # microns per pixel
LAMBDA_NM = 633       # illumination wavelength
ALPHA = 0.18          # refractive increment


def create_segmentation_overlay(image, masks):
    """
    Creates colored mask overlay for visualization
    """
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]

    for uid in unique_ids:
        mask = (masks == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0,255,0), 1)

    return overlay


def main():

    recon_root = settings.RECON_OUTPUT_FOLDER

    model = load_cellpose_model()

    folders = [f for f in recon_root.iterdir() if f.is_dir()]

    print(f"Found {len(folders)} reconstructed samples")

    all_records = []

    for folder in tqdm(folders):

        phase_png = folder / "phase.png"
        phase_npy = folder / "phase.npy"

        if not phase_png.exists() or not phase_npy.exists():
            print("Skipping (missing files):", folder.name)
            continue

        #print("Processing:", folder.name)

        # -----------------------------
        # Load images
        # -----------------------------

        image = cv2.imread(str(phase_png), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Failed to load PNG:", phase_png)
            continue

        phase_real = np.load(phase_npy)

        # smoothing for better segmentation
        image = cv2.GaussianBlur(image, (5,5), 0)

        # -----------------------------
        # Cellpose segmentation
        # -----------------------------

        masks = run_cellpose(image, model)

        if masks is None or np.max(masks) == 0:
            print("No cells detected")
            continue

        # -----------------------------
        # Feature extraction
        # -----------------------------

        df = extract_features_per_cell(
            phase_real,
            masks,
            PIX_SIZE,
            LAMBDA_NM,
            ALPHA
        )

        if df is None or len(df) == 0:
            print("No features extracted")
            continue

        # -----------------------------
        # Labels
        # -----------------------------

        label = "d" if "_d" in folder.name else "n"

        df["label"] = label
        df["sample"] = folder.name

        # save per-sample features
        df.to_csv(folder / "cell_features.csv", index=False)

        all_records.append(df)

        # -----------------------------
        # Save masks + visualization
        # -----------------------------

        np.save(folder / "cell_masks.npy", masks)

        overlay = create_segmentation_overlay(image, masks)
        cv2.imwrite(str(folder / "segmented.png"), overlay)

    # -----------------------------
    # Combine dataset
    # -----------------------------

    if len(all_records) == 0:
        print("No valid samples found.")
        return

    dataset = pd.concat(all_records, ignore_index=True)

    dataset_path = recon_root / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    print("Training dataset saved to:", dataset_path)


if __name__ == "__main__":
    main()