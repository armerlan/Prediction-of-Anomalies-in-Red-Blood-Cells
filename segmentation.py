import cv2
import numpy as np
import pandas as pd
from skimage import measure
import os, sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Force Cellpose to use bundled models
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = resource_path("models/cellpose")


def load_cellpose_model():
    from cellpose import models
    return models.Cellpose(model_type="cyto2")


# --------------------------------------------------
# Run segmentation
# --------------------------------------------------

def run_cellpose(image, model, diameter=150, flow_threshold=0.2, min_size=10):

    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        min_size=min_size
    )

    return masks


# --------------------------------------------------
# Feature extraction per cell
# --------------------------------------------------

def extract_features_per_cell(phase_real, masks, pix_size, lambda_nm, alpha_cm3_per_g):

    phase_real = phase_real.astype(np.float32)
    props = measure.regionprops(masks, intensity_image=phase_real)

    records = []

    for p in props:

        cid = p.label
        area_pixels = p.area
        area_um2 = area_pixels * (pix_size ** 2)

        perimeter = p.perimeter if p.perimeter > 0 else np.nan

        mean_phase = p.intensity_mean
        median_phase = np.median(phase_real[masks == cid])
        std_phase = np.std(phase_real[masks == cid])
        min_phase = np.min(phase_real[masks == cid])
        max_phase = np.max(phase_real[masks == cid])

        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else np.nan
        elongation = p.axis_major_length / p.axis_minor_length if p.axis_minor_length > 0 else np.nan

        centroid = np.round(p.centroid).astype(int)
        center_phase = phase_real[centroid[0], centroid[1]]

        d_value = center_phase - max_phase

        total_phase = np.sum(phase_real[masks == cid])
        pixel_area = pix_size ** 2
        lambda_um = lambda_nm * 1e-3

        MCH = (10 * lambda_um * total_phase * pixel_area) / (2 * np.pi * alpha_cm3_per_g)
        MCHSD = MCH / area_um2 if area_um2 > 0 else np.nan

        records.append({
            "area_um2": area_um2,
            "perimeter": perimeter,
            "circularity": circularity,
            "elongation": elongation,
            "mean_phase": mean_phase,
            "std_phase": std_phase,
            "min_phase": min_phase,
            "max_phase": max_phase,
            "MCH_surface_density": MCHSD,
            "optical_volume": mean_phase * area_um2,
        })

    return pd.DataFrame(records)

def create_segmentation_overlay(image, masks):
    """
    Creates colored mask overlay for visualization
    """
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]

    for uid in unique_ids:
        mask = (masks == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0,0,0), 3)

    return overlay