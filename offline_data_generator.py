# -*- coding: utf-8 -*-
"""
Offline Data Generation Script - Pre-samples pixel points from raw data, 
extracts (seq, img) pairs, and saves them as .npy files by category.



class_1_unknown:
    A subset of class1 pixels specified by equivHT coordinate files.
    Features originate from logic pixels, but labels are set to fill (class2) during training.
    Completely non-overlapping with class_1 pixels.

Prerequisites:
    Run main.py in online mode first to automatically call online_data_prepare.py 
    and generate equivHT coordinate files:
        python main.py --data_mode online
    Coordinate files are saved in: data/online_data/equiv_ht/{fold}/equiv_ht_coords_pre_aug.npy

    Alternatively, run online_data_prepare.py directly:
        python online_data_prepare.py --num_equiv_ht 50

Usage:
    # Use default parameters (equivHT coordinates loaded from data/online_data/equiv_ht/)
    python offline_data_generator.py \
        --mask /path/to/mask.npy \
        --data /path/to/data.npz

    # Custom parameters
    python offline_data_generator.py \
        --mask /path/to/mask.npy \
        --data /path/to/data.npz \
        --equiv_ht_dir data/online_data/equiv_ht \
        --samples_per_class 5000 \
        --val_ratio 0.2 \
        --num_equiv_ht 100

    # Disable equivHT (no class_1_unknown generated)
    python offline_data_generator.py \
        --mask /path/to/mask.npy \
        --data /path/to/data.npz \
        --equiv_ht_dir /nonexistent/path

Key Arguments:
    --equiv_ht_dir      Directory for equivHT coordinate files. Default: data/online_data/equiv_ht
                        Reads {equiv_ht_dir}/{fold}/equiv_ht_coords_pre_aug.npy for each fold.
                        If the file does not exist, class_1_unknown generation is skipped.
    --samples_per_class Total samples per category. Default: 5000 (train + val combined).
    --val_ratio         Validation set ratio. Default: 0.2.
    --num_equiv_ht      Maximum number of equivHT samples to extract per fold. Default: None (use all).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import random

from utils import clear_data


# -------------- Data Validation --------------

def validate_inputs(mask_path, data_path):
    """
    Load and validate mask and data, returning (label_raw, data_raw).
    """
    if not os.path.isfile(mask_path):
        sys.exit(f"[ERROR] mask file does not exist: {mask_path}")
    if not os.path.isfile(data_path):
        sys.exit(f"[ERROR] data file does not exist: {data_path}")

    print(f"Loading mask: {mask_path}")
    label_raw = np.load(mask_path)
    print(f"  shape={label_raw.shape}  dtype={label_raw.dtype}  unique={np.unique(label_raw)}")

    print(f"Loading data: {data_path}")
    data_raw = np.load(data_path)
    if isinstance(data_raw, np.lib.npyio.NpzFile):
        if 'data' not in data_raw.files:
            sys.exit(f"[ERROR] 'data' key not found in .npz file. Available keys: {data_raw.files}")
        data_raw = data_raw['data']
    print(f"  shape={data_raw.shape}  dtype={data_raw.dtype}  min={data_raw.min():.3f}  max={data_raw.max():.3f}")

    if label_raw.ndim != 2:
        sys.exit(f"[ERROR] mask should be 2D (H, W), but got ndim={label_raw.ndim}, shape={label_raw.shape}")
    if data_raw.ndim != 3:
        sys.exit(f"[ERROR] data should be 3D (T, H, W), but got ndim={data_raw.ndim}, shape={data_raw.shape}")

    print("[Validation] Basic mask and data validation passed")
    return label_raw, data_raw

# -------------- Utility Functions --------------

def generate_offsets(n: int):
    """Generate a list of offsets for an n×n neighborhood relative to the center (0,0), n must be odd"""
    if n % 2 == 0 or n < 1:
        raise ValueError("n must be a positive odd integer")
    half = n // 2
    offsets = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            offsets.append((dy, dx))
    return offsets


def extract_sequence(img3d, center, directions):
    """Extract neighborhood time series from 3D image (T, H, W) based on offset list, returns (N, N, T)"""
    grid = np.array([(center[0] + dx, center[1] + dy) for dx, dy in directions])
    temp = img3d[:, grid[:, 0], grid[:, 1]]
    neighborhood_size = int(np.sqrt(len(directions)))
    grid_sequence = temp.reshape(-1, neighborhood_size, neighborhood_size)
    grid_sequence = grid_sequence.transpose(1, 2, 0)
    return grid_sequence


def extract_patch(frame, center, patch_size):
    """
    Crop a patch_size×patch_size sub-image centered at 'center' from a single frame.
    Uses np.tile for tiling border handling (consistent with original logic).
    """
    frame_ = np.tile(frame, (3, 3))
    cx = center[0] + frame.shape[0]
    cy = center[1] + frame.shape[1]
    half = patch_size // 2
    odd = patch_size % 2
    sx, ex = cx - half, cx + half + odd
    sy, ey = cy - half, cy + half + odd
    return frame_[sx:ex, sy:ey]


def transform(matrix, angle=None, doflip=False, flip1=False):
    """Perform rotation and flipping on a 2D matrix"""
    if angle is not None:
        if angle == 90:
            matrix = np.rot90(matrix)
        elif angle == 180:
            matrix = np.rot90(matrix, k=2)
        elif angle == 270:
            matrix = np.rot90(matrix, k=3)
    if doflip:
        if flip1:
            matrix = np.flip(matrix, 1)
        else:
            matrix = np.flip(matrix, 0)
    return matrix


def transform_3d(matrix_3d, angle=None, doflip=False, flip1=False):
    """Perform identical rotation/flipping on each slice of a 3D matrix (D, H, W)"""
    transformed = [transform(matrix_3d[i], angle, doflip, flip1) for i in range(matrix_3d.shape[0])]
    try:
        return np.stack(transformed, axis=0)
    except ValueError:
        return matrix_3d


# -------------- Global Preprocessing (Crop, Cleaning) --------------

def global_preprocess(label_raw, data_raw, crop_h=(60, -50), crop_w=(90, -250)):
    """
    Perform global cropping and outlier cleaning on complete raw data (consistent with main.py).
    Crop before splitting to ensure parameters match original training.

    Returns (label_cropped, data_cropped)
    """
    label = label_raw.copy()
    label[(label == 3) | (label == 4)] = 3

    crop_h_start, crop_h_end = crop_h
    crop_w_start, crop_w_end = crop_w
    label = label[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

    data = data_raw.copy()
    data = data[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

    print(f"  Global crop: rows[{crop_h_start}:{crop_h_end}], cols[{crop_w_start}:{crop_w_end}]")
    print(f"  After crop - label: {label.shape}, data: {data.shape}")

    # Outlier cleaning
    for i in range(data.shape[0]):
        data[i] = clear_data(data[i])

    return label, data


# -------------- Spatial Splitting --------------

def split_fold(label_raw, data_raw, fold):
    """
    Split raw data into fold1 (top half) / fold2 (bottom half).
    Returns (label_fold, data_fold) corresponding to the specified fold.
    """
    height = label_raw.shape[0]
    split_height = height // 2

    if fold == 'fold1':
        label_fold = label_raw[:split_height, :]
        data_fold = data_raw[:, :split_height, :]
    elif fold == 'fold2':
        label_fold = label_raw[split_height:, :]
        data_fold = data_raw[:, split_height:, :]
    else:
        raise ValueError(f"fold must be 'fold1' or 'fold2', got '{fold}'")

    return label_fold, data_fold


# -------------- Data Preprocessing --------------

def preprocess_data(label_fold, data_fold, equiv_ht_coords_pre_aug=None):
    """
    Normalize and augment mask and data within a fold.
    If pre-augmentation equivHT coordinates are provided, create a boolean mask 
    and augment it synchronously with the labels.

    Args:
        label_fold: Labels for the fold (H_fold, W)
        data_fold: Data for the fold (T, H_fold, W)
        equiv_ht_coords_pre_aug: EquivHT coordinates in pre-augmentation space (N, 2), optional

    Returns:
        (data_np, normal_data_np, label_np, equiv_ht_mask)
        equiv_ht_mask: Augmented boolean mask (H_fold, W), or None if no equivHT
    """
    label = label_fold.copy()
    data = data_fold.copy()

    print(f"  Fold data - label: {label.shape}, data: {data.shape}")

    # Create equivHT boolean mask before augmentation
    equiv_ht_mask = None
    if equiv_ht_coords_pre_aug is not None and len(equiv_ht_coords_pre_aug) > 0:
        equiv_ht_mask = np.zeros_like(label, dtype=np.uint8)
        equiv_ht_mask[equiv_ht_coords_pre_aug[:, 0], equiv_ht_coords_pre_aug[:, 1]] = 1
        print(f"  equivHT mask created: {len(equiv_ht_coords_pre_aug)} pixels marked")

    # Normalization
    normal_data = data.copy().astype(np.float32)
    for i in range(normal_data.shape[0]):
        temp = normal_data[i]
        normal_data[i] = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)

    # Global image random augmentation
    angle = np.random.choice([0, 90, 180, 270])
    flip_h = np.random.rand() > 0.5
    flip_v = np.random.rand() > 0.5
    data = transform_3d(data, angle, flip_h, flip_v)
    normal_data = transform_3d(normal_data, angle, flip_h, flip_v)
    label = transform(label, angle, flip_h, flip_v)
    if equiv_ht_mask is not None:
        equiv_ht_mask = transform(equiv_ht_mask, angle, flip_h, flip_v)

    return data, normal_data, label, equiv_ht_mask


# -------------- Main Generation Logic --------------

def _extract_and_save(data_np, frame, directions, patch_size,
                      chosen_rows, chosen_cols, n_sample,
                      val_ratio, output_dir, fold, class_name):
    """
    Extract features and save immediately after train/val partitioning to save memory.

    Determines indices, pre-allocates arrays, fills them during extraction,
    and saves/releases immediately upon completion.
    """
    # Determine split first
    perm = np.random.permutation(n_sample)
    n_val = int(n_sample * val_ratio)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    # Get sample shape for pre-allocation
    sample_seq = extract_sequence(data_np, (chosen_rows[0], chosen_cols[0]), directions)
    sample_img = extract_patch(frame, (chosen_rows[0], chosen_cols[0]), patch_size)
    seq_shape = sample_seq.shape
    img_shape = sample_img.shape

    for phase, indices in [('train', train_indices), ('val', val_indices)]:
        n = len(indices)
        if n == 0:
            continue

        seqs = np.empty((n,) + seq_shape, dtype=np.float32)
        imgs = np.empty((n,) + img_shape, dtype=np.float32)

        for j, idx in enumerate(indices):
            center = (chosen_rows[idx], chosen_cols[idx])
            seqs[j] = extract_sequence(data_np, center, directions)
            imgs[j] = extract_patch(frame, center, patch_size)

        out_dir = os.path.join(output_dir, fold, phase)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'{class_name}_seqs.npy'), seqs)
        np.save(os.path.join(out_dir, f'{class_name}_imgs.npy'), imgs)
        print(f"  Saved {fold}/{phase}/{class_name}: seqs={seqs.shape}, imgs={imgs.shape}")

        del seqs, imgs

    print(f"  {class_name}: total={n_sample}, train={len(train_indices)}, val={len(val_indices)}")


def generate_fold(label_fold, data_fold, output_dir, fold,
                  samples_per_class, equiv_ht_coords_pre_aug,
                  neighborhood_size, patch_size, val_ratio=0.2):
    """
    Generate offline samples for a fold, containing three categories: class_1 / class_2 / class_1_unknown.

    class_1_unknown and class_1 pixels are completely non-overlapping and non-redundant.
    xtract and write iteratively; save and release memory immediately after each phase.
    """
    print(f"\n{'='*60}")
    print(f"Generating fold: {fold}")
    print(f"{'='*60}")

    data_np, normal_data_np, label_np, equiv_ht_mask = preprocess_data(
        label_fold, data_fold, equiv_ht_coords_pre_aug)
    height, width = label_np.shape
    directions = generate_offsets(neighborhood_size)

    print(f"  Data shape: {data_np.shape}, Label shape: {label_np.shape}")
    print(f"  Unique labels: {np.unique(label_np)}")

    # Take one frame for patch cropping
    frame_index = np.random.randint(0, normal_data_np.shape[0])
    frame = normal_data_np[frame_index]

    # Boundary filtering
    half = neighborhood_size // 2
    margin = max(half, patch_size // 2 + 1)

    # ---- All valid class1 pixels ----
    positions_cls1 = np.where(label_np == 1)
    valid_mask_cls1 = (
        (positions_cls1[0] > margin) & (positions_cls1[0] < height - margin) &
        (positions_cls1[1] > margin) & (positions_cls1[1] < width - margin)
    )
    cls1_valid_rows = positions_cls1[0][valid_mask_cls1]
    cls1_valid_cols = positions_cls1[1][valid_mask_cls1]
    total_cls1_valid = len(cls1_valid_rows)

    if total_cls1_valid == 0:
        print(f"  WARNING: class 1 has no valid pixels after boundary filtering.")
        del data_np, normal_data_np, label_np
        return

    # ---- Use equivHT mask to partition class_1_unknown and class_1 ----
    if equiv_ht_mask is not None:
        # Determination using augmented mask
        cls1_is_ht = equiv_ht_mask[cls1_valid_rows, cls1_valid_cols] == 1
        unknown_rows = cls1_valid_rows[cls1_is_ht]
        unknown_cols = cls1_valid_cols[cls1_is_ht]
        cls1_remaining_rows = cls1_valid_rows[~cls1_is_ht]
        cls1_remaining_cols = cls1_valid_cols[~cls1_is_ht]
    else:
        unknown_rows = np.array([], dtype=cls1_valid_rows.dtype)
        unknown_cols = np.array([], dtype=cls1_valid_cols.dtype)
        cls1_remaining_rows = cls1_valid_rows
        cls1_remaining_cols = cls1_valid_cols

    n_unknown = len(unknown_rows)
    total_cls1_remaining = len(cls1_remaining_rows)

    print(f"  class 1: total valid={total_cls1_valid}, "
          f"unknown={n_unknown}, remaining for class_1={total_cls1_remaining}")

    # ---- Extract and save class_1_unknown ----
    if n_unknown > 0:
        _extract_and_save(data_np, frame, directions, patch_size,
                          unknown_rows, unknown_cols, n_unknown,
                          val_ratio, output_dir, fold, 'class_1_unknown')
    else:
        print(f"  class_1_unknown: no equivHT pixels, skipping")

    # ---- Sample and save class_1 (from remaining pixels) ----
    n_sample_cls1 = min(samples_per_class, total_cls1_remaining)
    replace_cls1 = samples_per_class > total_cls1_remaining
    if replace_cls1:
        print(f"  class_1: only {total_cls1_remaining} remaining pixels, "
              f"will sample with replacement to reach {samples_per_class}")

    chosen_indices_cls1 = np.random.choice(total_cls1_remaining, size=n_sample_cls1, replace=replace_cls1)
    chosen_rows_cls1 = cls1_remaining_rows[chosen_indices_cls1]
    chosen_cols_cls1 = cls1_remaining_cols[chosen_indices_cls1]

    _extract_and_save(data_np, frame, directions, patch_size,
                      chosen_rows_cls1, chosen_cols_cls1, n_sample_cls1,
                      val_ratio, output_dir, fold, 'class_1')

    del chosen_rows_cls1, chosen_cols_cls1

    # ---- Sample and save class_2 ----
    positions_cls2 = np.where(label_np == 2)
    valid_mask_cls2 = (
        (positions_cls2[0] > margin) & (positions_cls2[0] < height - margin) &
        (positions_cls2[1] > margin) & (positions_cls2[1] < width - margin)
    )
    cls2_valid_rows = positions_cls2[0][valid_mask_cls2]
    cls2_valid_cols = positions_cls2[1][valid_mask_cls2]
    total_cls2_valid = len(cls2_valid_rows)

    if total_cls2_valid == 0:
        print(f"  WARNING: class 2 has no valid pixels after boundary filtering, skipping class_2.")
    else:
        n_sample_cls2 = min(samples_per_class, total_cls2_valid)
        replace_cls2 = samples_per_class > total_cls2_valid
        if replace_cls2:
            print(f"  class_2: only {total_cls2_valid} valid pixels, "
                  f"will sample with replacement to reach {samples_per_class}")

        chosen_indices_cls2 = np.random.choice(total_cls2_valid, size=n_sample_cls2, replace=replace_cls2)
        chosen_rows_cls2 = cls2_valid_rows[chosen_indices_cls2]
        chosen_cols_cls2 = cls2_valid_cols[chosen_indices_cls2]

        _extract_and_save(data_np, frame, directions, patch_size,
                          chosen_rows_cls2, chosen_cols_cls2, n_sample_cls2,
                          val_ratio, output_dir, fold, 'class_2')

    del data_np, normal_data_np, label_np
    print(f"fold={fold} done.")


# -------------- Entry Point --------------

def main():
    parser = argparse.ArgumentParser(description="Offline Data Generator (fold1/fold2 spatial partitioning)")
    parser.add_argument('--mask', type=str, required=True,
                        help='Absolute path to the mask label file (.npy)')
    parser.add_argument('--data', type=str, required=True,
                        help='Absolute path to the time-series data file (.npz or .npy)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data file directory/offline_data)')
    parser.add_argument('--equiv_ht_dir', type=str, default=None,
                        help='Directory for equivHT coordinate files (default: PROJECT_ROOT/data/online_data/equiv_ht)')
    parser.add_argument('--samples_per_class', type=int, default=5000,
                        help='Total samples per category for each fold (default: 5000)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation set to total samples (default: 0.2)')
    parser.add_argument('--num_equiv_ht', type=int, default=None,
                        help='Maximum number of equivHT samples to extract per fold (default: None, use all available)')
    parser.add_argument('--crop_h', type=str, default='60,-50',
                        help='Row cropping range start,end (default: 60,-50)')
    parser.add_argument('--crop_w', type=str, default='90,-250',
                        help='Column cropping range start,end (default: 90,-250)')
    parser.add_argument('--neighborhood_size', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    crop_h = tuple(int(x) for x in args.crop_h.split(','))
    crop_w = tuple(int(x) for x in args.crop_w.split(','))
    if len(crop_h) != 2 or len(crop_w) != 2:
        sys.exit("[ERROR] --crop_h and --crop_w should be in start,end format, e.g., 60,-50")

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'data', 'offline_data')

    if args.equiv_ht_dir is None:
        args.equiv_ht_dir = os.path.join(PROJECT_ROOT, 'data', 'online_data', 'equiv_ht')

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load and validate
    label_raw, data_raw = validate_inputs(args.mask, args.data)

    # Global crop and cleaning (consistent with main.py, crop before splitting)
    print("\n[Global Preprocess] Cropping and cleaning...")
    label_cropped, data_cropped = global_preprocess(label_raw, data_raw, crop_h, crop_w)
    del label_raw, data_raw

    # Generate separately for fold1 / fold2
    for fold in ['fold1', 'fold2']:
        label_fold, data_fold = split_fold(label_cropped, data_cropped, fold)
        print(f"\n[Fold] {fold}: label shape={label_fold.shape}, data shape={data_fold.shape}")

        # Load pre-augmentation equivHT coordinates for this fold
        equiv_ht_coords_path = os.path.join(args.equiv_ht_dir, fold, 'equiv_ht_coords_pre_aug.npy')
        equiv_ht_coords_pre_aug = None
        if os.path.isfile(equiv_ht_coords_path):
            equiv_ht_coords_pre_aug = np.load(equiv_ht_coords_path)
            print(f"  Loaded equivHT coords (pre-aug): {equiv_ht_coords_pre_aug.shape}")
            
            # Apply limitation on the number of equivHT pixels
            if args.num_equiv_ht is not None and args.num_equiv_ht < len(equiv_ht_coords_pre_aug):
                print(f"  Limiting equivHT coords from {len(equiv_ht_coords_pre_aug)} to {args.num_equiv_ht}")
                # Randomly select up to args.num_equiv_ht indices without replacement
                indices = np.random.choice(len(equiv_ht_coords_pre_aug), args.num_equiv_ht, replace=False)
                equiv_ht_coords_pre_aug = equiv_ht_coords_pre_aug[indices]
                
        else:
            print(f"  No equivHT coords file found at {equiv_ht_coords_path}, "
                  f"class_1_unknown will not be generated")

        generate_fold(
            label_fold=label_fold,
            data_fold=data_fold,
            output_dir=args.output_dir,
            fold=fold,
            samples_per_class=args.samples_per_class,
            equiv_ht_coords_pre_aug=equiv_ht_coords_pre_aug,
            neighborhood_size=args.neighborhood_size,
            patch_size=args.patch_size,
            val_ratio=args.val_ratio,
        )

    print("\nAll done.")


if __name__ == '__main__':
    main()