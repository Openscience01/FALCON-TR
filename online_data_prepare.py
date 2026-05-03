# -*- coding: utf-8 -*-
"""
Online Data Preparation Module - Provides data interface for online training with fold splitting

Usage (import as a module):
    from online_data_prepare import prepare_fold_data
    result = prepare_fold_data(...)

Usage (run independently to generate coordinate files):
    python online_data_prepare.py
    python online_data_prepare.py --num_equiv_ht 50 --val_ratio 0.2
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np

from offline_data_generator import global_preprocess, split_fold, transform, transform_3d
from utils import clear_data


def prepare_fold_data(base_dir, fold, val_ratio=0.2, num_equiv_ht=50,
                      time_length=1000, margin=1, seed=42):
    """
    Prepares online training data for a specified fold.

    Parameters:
        base_dir: Directory containing mask.npy and data.npz
        fold: 'fold1' or 'fold2'
        val_ratio: Validation set ratio (default: 0.2)
        num_equiv_ht: Number of equivHT samples to extract from class1 (default: 50)
        time_length: Temporal length (default: 1000)
        margin: Number of pixels to exclude at boundaries (default: 1, consistent with original _sample_points)
        seed: Random seed (default: 42)

    Returns:
        data_np: Raw temporal data for the fold (T, H_fold, W)
        normal_data_np: Normalized temporal data (T, H_fold, W)
        label_np: Labels (H_fold, W)
        equiv_ht_coords: equivHT pixel coordinates in augmented space (N, 2), each row (row, col)
        train_coords: {'class1': (rows, cols), 'class2': (rows, cols)}
        val_coords: {'class1': (rows, cols), 'class2': (rows, cols)}
        equiv_ht_coords_pre_aug: equivHT coordinates in pre-augmentation space (N, 2), for offline data generation use
    """
    np.random.seed(seed)

    # ---- 1. Load, Crop, and Clean ----
    label_raw = np.load(os.path.join(base_dir, 'mask.npy'))
    label_raw[(label_raw == 3) | (label_raw == 4)] = 3

    data_raw = np.load(os.path.join(base_dir, 'data.npz'))['data']
    data_raw = data_raw[:time_length]

    label_cropped, data_cropped = global_preprocess(label_raw, data_raw)
    del label_raw, data_raw

    # ---- 2. Spatial Fold Splitting ----
    label_fold, data_fold = split_fold(label_cropped, data_cropped, fold)
    del label_cropped, data_cropped

    height_pre, width_pre = label_fold.shape

    # ---- 3. Before Augmentation: Extract equivHT (Coordinates in pre-augmentation space) ----
    positions_cls1 = np.where(label_fold == 1)
    valid_cls1 = (
        (positions_cls1[0] > margin) & (positions_cls1[0] < height_pre - margin) &
        (positions_cls1[1] > margin) & (positions_cls1[1] < width_pre - margin)
    )
    cls1_rows_pre = positions_cls1[0][valid_cls1]
    cls1_cols_pre = positions_cls1[1][valid_cls1]
    n_cls1 = len(cls1_rows_pre)

    n_ht = min(num_equiv_ht, n_cls1)
    ht_indices = np.random.choice(n_cls1, size=n_ht, replace=False)
    equiv_ht_coords_pre_aug = np.stack([cls1_rows_pre[ht_indices], cls1_cols_pre[ht_indices]], axis=1)

    # Create equivHT boolean mask (same shape as label) to mark which class1 pixels are equivHT
    equiv_ht_mask = np.zeros_like(label_fold, dtype=np.uint8)
    equiv_ht_mask[equiv_ht_coords_pre_aug[:, 0], equiv_ht_coords_pre_aug[:, 1]] = 1

    # ---- 4. Normalization + Augmentation (Synchronized mask transformation) ----
    data_np = data_fold.copy()
    normal_data_np = data_fold.copy().astype(np.float32)
    for i in range(normal_data_np.shape[0]):
        temp = normal_data_np[i]
        normal_data_np[i] = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)

    angle = np.random.choice([0, 90, 180, 270])
    flip_h = np.random.rand() > 0.5
    flip_v = np.random.rand() > 0.5
    data_np = transform_3d(data_np, angle, flip_h, flip_v)
    normal_data_np = transform_3d(normal_data_np, angle, flip_h, flip_v)
    label_np = transform(label_fold, angle, flip_h, flip_v)
    equiv_ht_mask = transform(equiv_ht_mask, angle, flip_h, flip_v)
    del data_fold, label_fold

    height, width = label_np.shape

    # ---- 5. Extract equivHT Coordinates from Augmented Mask ----
    aug_ht_positions = np.where(equiv_ht_mask == 1)
    equiv_ht_coords = np.stack([aug_ht_positions[0], aug_ht_positions[1]], axis=1)

    # ---- 6. Augmented Space: Find valid pixels, exclude equivHT ----
    positions_cls1 = np.where(label_np == 1)
    valid_cls1 = (
        (positions_cls1[0] > margin) & (positions_cls1[0] < height - margin) &
        (positions_cls1[1] > margin) & (positions_cls1[1] < width - margin)
    )
    cls1_rows = positions_cls1[0][valid_cls1]
    cls1_cols = positions_cls1[1][valid_cls1]
    n_cls1 = len(cls1_rows)

    positions_cls2 = np.where(label_np == 2)
    valid_cls2 = (
        (positions_cls2[0] > margin) & (positions_cls2[0] < height - margin) &
        (positions_cls2[1] > margin) & (positions_cls2[1] < width - margin)
    )
    cls2_rows = positions_cls2[0][valid_cls2]
    cls2_cols = positions_cls2[1][valid_cls2]
    n_cls2 = len(cls2_rows)

    # Exclude equivHT from class1 (determined using augmented mask)
    cls1_not_ht = equiv_ht_mask[cls1_rows, cls1_cols] == 0
    cls1_remaining_rows = cls1_rows[cls1_not_ht]
    cls1_remaining_cols = cls1_cols[cls1_not_ht]
    n_cls1_remaining = len(cls1_remaining_rows)

    print(f"  [{fold}] class1 valid={n_cls1}, equivHT={len(equiv_ht_coords)}, remaining={n_cls1_remaining}")
    print(f"  [{fold}] class2 valid={n_cls2}")

    # ---- 7. Randomly split train/val coordinates according to val_ratio ----
    def _split_coords(rows, cols, val_ratio):
        n = len(rows)
        perm = np.random.permutation(n)
        n_val = int(n * val_ratio)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return {
            'train': (rows[train_idx], cols[train_idx]),
            'val': (rows[val_idx], cols[val_idx]),
        }

    cls1_split = _split_coords(cls1_remaining_rows, cls1_remaining_cols, val_ratio)
    cls2_split = _split_coords(cls2_rows, cls2_cols, val_ratio)

    train_coords = {
        'class1': cls1_split['train'],
        'class2': cls2_split['train'],
    }
    val_coords = {
        'class1': cls1_split['val'],
        'class2': cls2_split['val'],
    }

    print(f"  [{fold}] train: class1={len(train_coords['class1'][0])}, "
          f"class2={len(train_coords['class2'][0])}")
    print(f"  [{fold}] val:   class1={len(val_coords['class1'][0])}, "
          f"class2={len(val_coords['class2'][0])}")

    return data_np, normal_data_np, label_np, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug


def save_coords(base_dir, fold, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug):
    """Saves coordinate files to data/online_data/equiv_ht/{fold}/"""
    out_dir = os.path.join(base_dir, 'equiv_ht', fold)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'equiv_ht_coords.npy'), equiv_ht_coords)
    np.save(os.path.join(out_dir, 'equiv_ht_coords_pre_aug.npy'), equiv_ht_coords_pre_aug)

    for phase, coords in [('train', train_coords), ('val', val_coords)]:
        for cls_name in ['class1', 'class2']:
            rows, cols = coords[cls_name]
            arr = np.stack([rows, cols], axis=1)
            np.save(os.path.join(out_dir, f'{phase}_coords_{cls_name}.npy'), arr)

    print(f"  [{fold}] coords saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Online Data Preparation - Generates fold coordinate files")
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Directory containing mask.npy and data.npz')
    parser.add_argument('--num_equiv_ht', type=int, default=50,
                        help='Number of equivHT for each fold (default: 50)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--time_length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.join(PROJECT_ROOT, 'data', 'online_data')

    for fold in ['fold1', 'fold2']:
        print(f"\n{'='*60}")
        print(f"Preparing fold: {fold}")
        print(f"{'='*60}")

        data_np, normal_data_np, label_np, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug = \
            prepare_fold_data(
                base_dir, fold,
                val_ratio=args.val_ratio,
                num_equiv_ht=args.num_equiv_ht,
                time_length=args.time_length,
                seed=args.seed,
            )

        save_coords(base_dir, fold, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug)

    print("\nAll done.")


if __name__ == '__main__':
    main()
    