import numpy as np
from numpy.fft import fft2, fftshift
import os, sys
from application.logger import logger

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model = None

prediction_scaler = None
prediction_clf = None
prediction_features = None
prediction_threshold = None

def get_model():
    global model
    if model is None:
        from backend.segmentation import load_cellpose_model
        logger.info("Loading Cellpose model")
        model = load_cellpose_model()
        logger.info("Cellpose model loaded successfully")
    return model

def get_prediction_threshold():

    global prediction_threshold

    if prediction_threshold is None:
        preload_prediction_models()

    return prediction_threshold

def preload_prediction_models():

    global prediction_scaler
    global prediction_clf
    global prediction_features
    global prediction_threshold

    if prediction_scaler is None:

        import joblib

        logger.info("Loading prediction model")

        prediction_scaler = joblib.load(
            resource_path("models/scaler.pkl")
        )

        prediction_clf = joblib.load(
            resource_path("models/xgboost_trained.pkl")
        )

        prediction_features = joblib.load(
            resource_path("models/features.pkl")
        )

        prediction_threshold = joblib.load(
            resource_path("models/threshold.pkl")
        )

        logger.info("Prediction models loaded successfully")

def create_phase_visualization(
        phase,
        cmap="turbo",
        label="Phase (rad)"
    ):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np

    vmin = -np.pi
    vmax = np.pi

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)

    im = ax.imshow(
        phase,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(label)
    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    img = buf[:, :, :3]

    plt.close(fig)

    # RGB -> BGR for OpenCV/Qt consistency
    img = img[:, :, ::-1].copy()

    return img

def generate_reconstructions(obj_image, ref_images, peak_y, peak_x, dx=50, dy=50, progress_callback = None):
    from backend.reconstruction import reconstruct_from_hologram
    import cv2

    reconstructions = []

    logger.info(
        f"Generating reconstructions for peak "
        f"({peak_x}, {peak_y}) using {len(ref_images)} references"
    )

    obj_field = reconstruct_from_hologram(obj_image, peak_y, peak_x, dx, dy)

    for i, ref in enumerate(ref_images):

        ref_field = reconstruct_from_hologram(ref, peak_y, peak_x, dx, dy)
        total = len(ref_images)

        field = obj_field / (ref_field + 1e-8)

        phase = np.angle(field)

        # # normalize for GUI
        # phase_normalized = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        # phase_display = cv2.applyColorMap(phase_normalized, cv2.COLORMAP_TURBO)

        phase_display = create_phase_visualization(
            phase,
            cmap="turbo",
            label="Phase (rad)"
        )

        reconstructions.append({
            "id": i,
            "phase": phase,
            "phase_display": phase_display
        })

        if progress_callback:

            should_continue = progress_callback(i + 1, total)

            if should_continue is False:
                logger.info("Reconstruction cancelled")
                return None

    # logger.info(
    #     f"Generated {len(reconstructions)} reconstructions"
    # )

    return reconstructions

def compute_fft_magnitude(image):
    F = fftshift(fft2(image))
    mag = np.log(np.abs(F) + 1)
    return mag

def analyze_phase(phase, pix_size, lambda_nm, alpha):

    from backend.segmentation import run_cellpose, extract_features_per_cell

    model = get_model()

    masks = run_cellpose(phase, model)

    df = extract_features_per_cell(
        phase, masks,
        pix_size=pix_size,
        lambda_nm=lambda_nm,
        alpha_cm3_per_g=alpha
    )

    return masks, df

def analyze_reconstruction(phase_image, progress_callback):
    import cv2

    from backend.segmentation import run_cellpose, extract_features_per_cell, create_segmentation_overlay

    logger.info("Starting reconstruction analysis")

    if progress_callback.is_cancelled():
        return

    progress_callback.emit(10, "Preparing image...")

    img_for_seg = phase_image.copy()

    img_for_seg = (img_for_seg - np.min(img_for_seg)) / (np.max(img_for_seg) - np.min(img_for_seg) + 1e-8)
    img_for_seg = (img_for_seg * 255).astype(np.uint8)

    # display_image = cv2.applyColorMap(
    #     img_for_seg,
    #     cv2.COLORMAP_TURBO
    # )

    display_phase_only = cv2.applyColorMap(
        img_for_seg,
        cv2.COLORMAP_TURBO
    )

    display_image = create_phase_visualization(
        phase_image,
        cmap="turbo",
        label="Phase (rad)"
    )

    img_for_seg = cv2.GaussianBlur(img_for_seg, (5,5), 0)

    if progress_callback.is_cancelled():
        return

    progress_callback.emit(30, "Running Cellpose segmentation...")

    model = get_model()
    masks = run_cellpose(img_for_seg, model)

    if progress_callback.is_cancelled():
        return

    if masks is None or np.max(masks) == 0:
        logger.warning("No cells detected during segmentation")
    
    logger.info("Cellpose segmentation complete")

    if progress_callback.is_cancelled():
        return

    progress_callback.emit(70, "Extracting features...")

    df = extract_features_per_cell(
        phase_real=phase_image,
        masks=masks,
        pix_size=0.0405,
        lambda_nm=633,
        alpha_cm3_per_g=0.18
    )

    if df is None or len(df) == 0:
        logger.warning("No features extracted from segmented cells")
        return None, [], {
            "cell_count": 0,
            "statistics": {}
        }

    df = df.fillna(0)

    table_data = df.round(3).to_dict(orient="records")

    stats_data = {
        "cell_count": len(df),
        "statistics": {
            "Mean": df.mean(numeric_only=True).round(3).to_dict(),
            "Median": df.median(numeric_only=True).round(3).to_dict(),
            "Std Dev": df.std(numeric_only=True).round(3).to_dict(),
            "Min": df.min(numeric_only=True).round(3).to_dict(),
            "Max": df.max(numeric_only=True).round(3).to_dict(),
        }
    }

    if progress_callback.is_cancelled():
        return

    segmented_image = create_segmentation_overlay(display_image, masks)

    segmented_image = create_segmentation_overlay(
        display_phase_only,
        masks
    )

    if progress_callback.is_cancelled():
        return
    
    progress_callback.emit(100, "Done")

    # logger.info(
    #     f"Analysis complete: {size} cells analyzed"
    # )

    return segmented_image, table_data, stats_data

def predict_from_table(table_data, threshold = None):
    import pandas as pd

    global prediction_scaler
    global prediction_clf
    global prediction_features
    global prediction_threshold

    logger.info("Running prediction inference")

    if prediction_scaler is None:
        preload_prediction_models()

    scaler = prediction_scaler
    clf = prediction_clf
    features = prediction_features
    
    if threshold is None:
        threshold = prediction_threshold

    df = pd.DataFrame(table_data)

    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"Missing prediction features: {missing}")

    if df.empty:
        logger.warning("Prediction attempted with empty table")
        return "No Data", 0.0

    df = df.fillna(0)
    X = df[features].values
    X_scaled = scaler.transform(X)

    cell_probs = clf.predict_proba(X_scaled)[:, 1]

    image_prob = cell_probs.mean()
    image_pred = int(image_prob > threshold)

    label_map = {0: "Normal", 1: "Diabetic"}
    prediction = label_map[image_pred]

    confidence = image_prob * 100

    if prediction == "Normal":
        confidence = 100 - confidence

    return prediction, confidence