import os
import numpy as np
import pandas as pd
import cv2
from itertools import product
from pathlib import Path
from tqdm import tqdm
from backend.config import settings
from backend.reconstruction import reconstruct_from_hologram

# -----------------------------
# USER PATHS
# -----------------------------

OBJ_DIR = settings.DATA_FOLDER / "obj"
REF_DIR = settings.DATA_FOLDER / "ref"
OUTPUT_DIR = settings.RECON_OUTPUT_FOLDER

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_group_id(filename):
    """
    Example:
    a1_1_d.png -> 1_d
    r3_2_n.png -> 2_n
    """

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    return f"{parts[1]}_{parts[2]}"


def get_label(filename):
    """
    Extract diabetic/normal label
    """

    name = os.path.splitext(filename)[0]
    return name.split("_")[-1]


def reconstruct_phase(obj_path, ref_path):
    """
    Holographic reconstruction pipeline
    """

    obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # reconstruct fields
    obj_field = reconstruct_from_hologram(obj)
    ref_field = reconstruct_from_hologram(ref)

    # isolate object wave
    pure_object = obj_field / (ref_field + 1e-8)

    phase = np.angle(pure_object)

    return phase


def save_phase_outputs(phase, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "phase.npy"), phase)

    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
    phase_uint8 = phase_norm.astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, "phase.png"), phase_uint8)


# -----------------------------
# LOAD FILES
# -----------------------------

obj_files = [f.name for f in OBJ_DIR.iterdir() if f.name.startswith("a")]
ref_files = [f.name for f in REF_DIR.iterdir() if f.name.startswith("r")]

print(f"Found {len(obj_files)} object images")
print(f"Found {len(ref_files)} reference images")

TARGET_OBJECT = "a39_2_d.bmp"
TARGET_GROUP = "2_d"

# -----------------------------
# GROUP FILES
# -----------------------------

obj_groups = {}
ref_groups = {}

for f in obj_files:
    gid = get_group_id(f)
    obj_groups.setdefault(gid, []).append(f)

for f in ref_files:
    gid = get_group_id(f)
    ref_groups.setdefault(gid, []).append(f)

# -----------------------------
# MATCH PAIRS
# -----------------------------

metadata = []

objs = [TARGET_OBJECT]
refs = ref_groups.get(TARGET_GROUP, [])

print(f"\nProcessing only {TARGET_OBJECT}")
print(f"{len(refs)} references found")

for ref_name in tqdm(refs):

    obj_path = os.path.join(OBJ_DIR, TARGET_OBJECT)
    ref_path = os.path.join(REF_DIR, ref_name)

    pair_name = f"{Path(TARGET_OBJECT).stem}__{Path(ref_name).stem}"
    save_dir = os.path.join(OUTPUT_DIR, pair_name)

    # reconstruction
    phase = reconstruct_phase(obj_path, ref_path)

    # save
    save_phase_outputs(phase, save_dir)

    metadata.append({
        "obj_image": TARGET_OBJECT,
        "ref_image": ref_name,
        "set_id": TARGET_GROUP,
        "label": get_label(TARGET_OBJECT),
        "output_path": save_dir
    })

# -----------------------------
# SAVE METADATA
# -----------------------------

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata2.csv"), index=False)

print("\nDataset generation complete.")
print(f"{len(metadata)} reconstructions saved.")