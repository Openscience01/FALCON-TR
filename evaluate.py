# -*- coding: utf-8 -*-
"""
Cross-evaluation program based on offline data

Two parts of evaluation:
    1. Standard metrics: Each of the two fold models predicts on the train+val data of both folds, 
       outputting Precision/Recall/Accuracy.
    2. HT Detection Rate: Both fold models detect the class_1_unknown data from fold1, 
       outputting the detection rate.

Prerequisites:
    Run offline_data_generator.py first to generate offline data.

Usage:
    python evaluate.py
    python evaluate.py --num_ref_points 5
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch

from model import CTE_3D, HSE, SiameseNet


# -------------- Data Loading --------------

def discover_folds_data(data_dir):
    """Scan offline data directory and return the list of existing folds."""
    folds = []
    if not os.path.isdir(data_dir):
        return folds
    for name in sorted(os.listdir(data_dir)):
        if name.startswith('fold') and os.path.isdir(os.path.join(data_dir, name)):
            folds.append(name)
    return folds


def discover_folds_model(weights_dir, model_subdir, model_filename):
    """Scan weights directory and return the list of existing (fold name, model path)."""
    results = []
    if not os.path.isdir(weights_dir):
        return results
    for name in sorted(os.listdir(weights_dir)):
        if name.startswith('fold') and os.path.isdir(os.path.join(weights_dir, name)):
            model_path = os.path.join(weights_dir, name, model_subdir, model_filename)
            if os.path.isfile(model_path):
                results.append((name, model_path))
    return results


def load_fold_data(data_dir, fold):
    """
    Load offline data for a specified fold (merge train + val).
    Returns:
        class_data: {1: {'seqs', 'imgs'}, 2: {'seqs', 'imgs'}}
        unknown_data: {'seqs', 'imgs'} or None
    """
    class_data = {}
    unknown_data = None

    for phase in ['train', 'val']:
        fold_dir = os.path.join(data_dir, fold, phase)
        if not os.path.isdir(fold_dir):
            continue

        for cls in [1, 2]:
            seq_path = os.path.join(fold_dir, f'class_{cls}_seqs.npy')
            img_path = os.path.join(fold_dir, f'class_{cls}_imgs.npy')
            if not os.path.isfile(seq_path) or not os.path.isfile(img_path):
                continue

            seqs = np.load(seq_path)
            imgs = np.load(img_path)

            if cls not in class_data:
                class_data[cls] = {'seqs': seqs, 'imgs': imgs}
            else:
                class_data[cls]['seqs'] = np.concatenate([class_data[cls]['seqs'], seqs], axis=0)
                class_data[cls]['imgs'] = np.concatenate([class_data[cls]['imgs'], imgs], axis=0)

        unk_seq_path = os.path.join(fold_dir, 'class_1_unknown_seqs.npy')
        unk_img_path = os.path.join(fold_dir, 'class_1_unknown_imgs.npy')
        if os.path.isfile(unk_seq_path) and os.path.isfile(unk_img_path):
            unk_seqs = np.load(unk_seq_path)
            unk_imgs = np.load(unk_img_path)
            if unknown_data is None:
                unknown_data = {'seqs': unk_seqs, 'imgs': unk_imgs}
            else:
                unknown_data['seqs'] = np.concatenate([unknown_data['seqs'], unk_seqs], axis=0)
                unknown_data['imgs'] = np.concatenate([unknown_data['imgs'], unk_imgs], axis=0)

    return class_data, unknown_data


# -------------- Pair Construction --------------

def build_pairs(class_data, ref_idx=None):
    """Select a reference point from class_1 and iterate through class_1 + class_2 to build evaluation pairs."""
    ref_seqs = class_data[1]['seqs']
    ref_imgs = class_data[1]['imgs']

    if ref_idx is None:
        ref_idx = np.random.randint(0, ref_seqs.shape[0])
    ref_seq = ref_seqs[ref_idx]
    ref_img = ref_imgs[ref_idx]

    helper_idx = np.random.randint(0, class_data[2]['seqs'].shape[0])
    helper_seq = class_data[2]['seqs'][helper_idx]
    helper_img = class_data[2]['imgs'][helper_idx]

    helper_seq_pair = np.stack([ref_seq, helper_seq], axis=0)
    helper_img_pair = np.stack([ref_img, helper_img], axis=0)

    pairs = []
    for cls in [1, 2]:
        seqs = class_data[cls]['seqs']
        imgs = class_data[cls]['imgs']
        for i in range(seqs.shape[0]):
            seq_pair = np.stack([ref_seq, seqs[i]], axis=0)
            img_pair = np.stack([ref_img, imgs[i]], axis=0)
            pairs.append((seq_pair, img_pair, cls))

    return pairs, (helper_seq_pair, helper_img_pair)


def build_unknown_pairs(unknown_data, class_data, ref_idx=None):
    """Build evaluation pairs for class_1_unknown: class_1 reference point + unknown samples."""
    ref_seqs = class_data[1]['seqs']
    ref_imgs = class_data[1]['imgs']

    if ref_idx is None:
        ref_idx = np.random.randint(0, ref_seqs.shape[0])
    ref_seq = ref_seqs[ref_idx]
    ref_img = ref_imgs[ref_idx]

    helper_idx = np.random.randint(0, class_data[2]['seqs'].shape[0])
    helper_seq = class_data[2]['seqs'][helper_idx]
    helper_img = class_data[2]['imgs'][helper_idx]

    helper_seq_pair = np.stack([ref_seq, helper_seq], axis=0)
    helper_img_pair = np.stack([ref_img, helper_img], axis=0)

    unk_seqs = unknown_data['seqs']
    unk_imgs = unknown_data['imgs']

    pairs = []
    for i in range(unk_seqs.shape[0]):
        seq_pair = np.stack([ref_seq, unk_seqs[i]], axis=0)
        img_pair = np.stack([ref_img, unk_imgs[i]], axis=0)
        pairs.append((seq_pair, img_pair))

    return pairs, (helper_seq_pair, helper_img_pair)


# -------------- Batch Inference --------------

def batch_inference(model, pairs, helper_pair, batch_size, device):
    """Batch inference for standard pairs, returns (true_labels, pred_labels)."""
    model.eval()
    true_labels = []
    pred_labels = []
    helper_seq, helper_img = helper_pair

    n_pairs = len(pairs)
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_pairs = pairs[start:end]
        current_bs = end - start

        batch_seqs = []
        batch_imgs = []
        batch_true = []

        for seq_pair, img_pair, true_cls in batch_pairs:
            batch_seqs.append(seq_pair)
            batch_imgs.append(img_pair)
            batch_true.append(true_cls)
            batch_seqs.append(helper_seq)
            batch_imgs.append(helper_img)

        batch_seqs = np.stack(batch_seqs, axis=0)
        batch_imgs = np.stack(batch_imgs, axis=0)

        data = {
            'imgs': torch.tensor(batch_imgs).float().to(device),
            'seqs': torch.tensor(batch_seqs).float().to(device),
        }

        with torch.no_grad():
            output, _ = model(data)

        preds = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(current_bs):
            true_labels.append(batch_true[i])
            pred_labels.append(1 if preds[i * 2] == 1 else 2)

    return true_labels, pred_labels


def batch_inference_ht(model, pairs, helper_pair, batch_size, device):
    """Batch inference for HT pairs, returns whether each sample was detected (pred=class_1)."""
    model.eval()
    pred_detected = []
    helper_seq, helper_img = helper_pair

    n_pairs = len(pairs)
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_pairs = pairs[start:end]
        current_bs = end - start

        batch_seqs = []
        batch_imgs = []

        for seq_pair, img_pair in batch_pairs:
            batch_seqs.append(seq_pair)
            batch_imgs.append(img_pair)
            batch_seqs.append(helper_seq)
            batch_imgs.append(helper_img)

        batch_seqs = np.stack(batch_seqs, axis=0)
        batch_imgs = np.stack(batch_imgs, axis=0)

        data = {
            'imgs': torch.tensor(batch_imgs).float().to(device),
            'seqs': torch.tensor(batch_seqs).float().to(device),
        }

        with torch.no_grad():
            output, _ = model(data)

        preds = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(current_bs):
            pred_detected.append(preds[i * 2] == 1)

    return pred_detected


# -------------- Metric Calculation --------------

def calculate_metrics(true_labels, pred_labels):
    """Calculate Precision / Recall / Accuracy"""
    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)
    num_classes = 2
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_arr, pred_arr):
        confusion[int(t) - 1, int(p) - 1] += 1

    total = confusion.sum()
    accuracy = np.diag(confusion).sum() / (total + 1e-10)

    precision = []
    recall = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision.append(tp / (tp + fp + 1e-10))
        recall.append(tp / (tp + fn + 1e-10))

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}



def evaluate_fold(model, class_data, unknown_data, batch_size, device, num_ref_points):
    """
    Evaluate a single model's performance on a single fold's data.
    Returns (metrics, ht_metrics_dict)
    """
    if len(class_data) < 2:
        return None, None

    n_available = class_data[1]['seqs'].shape[0]
    actual_num_ref = min(num_ref_points, n_available)
    ref_indices = np.random.choice(n_available, size=actual_num_ref, replace=False)

    # ---- 1. Standard Evaluation (class_1 + class_2) ----
    sample_true = []
    sample_global_idx = {}
    for cls in [1, 2]:
        n = class_data[cls]['seqs'].shape[0]
        for i in range(n):
            gi = len(sample_true)
            sample_true.append(cls)
            sample_global_idx[(cls, i)] = gi

    n_samples = len(sample_true)
    sample_votes = [[] for _ in range(n_samples)]

    for ref_idx in ref_indices:
        pairs, helper_pair = build_pairs(class_data, ref_idx=ref_idx)
        _, pred_labels = batch_inference(model, pairs, helper_pair, batch_size, device)

        pair_i = 0
        for cls in [1, 2]:
            n = class_data[cls]['seqs'].shape[0]
            for i in range(n):
                gi = sample_global_idx[(cls, i)]
                sample_votes[gi].append(pred_labels[pair_i])
                pair_i += 1

    final_pred = []
    for votes in sample_votes:
        arr = np.array(votes)
        final_pred.append(1 if (arr == 1).sum() >= (arr == 2).sum() else 2)

    metrics = calculate_metrics(sample_true, final_pred)

    # ---- 2. HT Mixed Evaluation (class_1_unknown treated as positive, class_2 as negative) ----
    ht_metrics = None
    if unknown_data is not None and 2 in class_data:
        n_unk = unknown_data['seqs'].shape[0]
        n_cls2 = class_data[2]['seqs'].shape[0]

        unk_votes = [[] for _ in range(n_unk)]
        cls2_votes = [[] for _ in range(n_cls2)]

        for ref_idx in ref_indices:
            ref_seq = class_data[1]['seqs'][ref_idx]
            ref_img = class_data[1]['imgs'][ref_idx]
            
            # Build helper_pair (used for HSE module etc.)
            helper_idx = np.random.randint(0, n_cls2)
            helper_seq_pair = np.stack([ref_seq, class_data[2]['seqs'][helper_idx]], axis=0)
            helper_img_pair = np.stack([ref_img, class_data[2]['imgs'][helper_idx]], axis=0)
            helper_pair = (helper_seq_pair, helper_img_pair)

            # Build test pairs: class_1_unknown (Label=1)
            unk_pairs = []
            for i in range(n_unk):
                seq_p = np.stack([ref_seq, unknown_data['seqs'][i]], axis=0)
                img_p = np.stack([ref_img, unknown_data['imgs'][i]], axis=0)
                unk_pairs.append((seq_p, img_p, 1))

            # Build test pairs: class_2 (Label=2)
            cls2_pairs = []
            for i in range(n_cls2):
                seq_p = np.stack([ref_seq, class_data[2]['seqs'][i]], axis=0)
                img_p = np.stack([ref_img, class_data[2]['imgs'][i]], axis=0)
                cls2_pairs.append((seq_p, img_p, 2))

            # Merge and use standardized batch_inference for inference
            all_ht_pairs = unk_pairs + cls2_pairs
            _, preds = batch_inference(model, all_ht_pairs, helper_pair, batch_size, device)

            # Collect voting results
            for i in range(n_unk):
                unk_votes[i].append(preds[i])
            for i in range(n_cls2):
                cls2_votes[i].append(preds[n_unk + i])

        # Parse final voting results (majority vote)
        final_unk_preds = [1 if (np.array(v) == 1).sum() >= (np.array(v) == 2).sum() else 2 for v in unk_votes]
        final_cls2_preds = [1 if (np.array(v) == 1).sum() >= (np.array(v) == 2).sum() else 2 for v in cls2_votes]

        # Calculate metrics
        TP = sum(1 for p in final_unk_preds if p == 1)
        FN = sum(1 for p in final_unk_preds if p == 2)
        FP = sum(1 for p in final_cls2_preds if p == 1)
        TN = sum(1 for p in final_cls2_preds if p == 2)

        ht_dr = TP / (TP + FN + 1e-10)
        ht_fpr = FP / (FP + TN + 1e-10)
        ht_acc = (TP + TN) / (TP + FN + FP + TN + 1e-10)

        ht_metrics = {
            'detection_rate': ht_dr,
            'fpr': ht_fpr,
            'accuracy': ht_acc
        }

    return metrics, ht_metrics


def print_results(results, ht_results, model_folds, data_folds):
    """Print evaluation results"""
    col_w = 12

    # --- Standard Metrics ---
    for metric_name, key in [('Logic Precision', 'precision'), ('Logic Recall', 'recall'),
                              ('Fill Precision', 'precision'), ('Fill Recall', 'recall')]:
        idx = 0 if 'Logic' in metric_name else 1
        print(f"\n{metric_name}")
        header = f"{'Model':<12}"
        for df in data_folds:
            header += f"| {df:>{col_w}} "
        print(header)
        print("-" * (12 + (col_w + 3) * len(data_folds)))
        for mf in model_folds:
            row = f"{mf:<12}"
            for df in data_folds:
                k = (mf, df)
                if k in results:
                    row += f"| {results[k][key][idx]:>{col_w}.4f} "
                else:
                    row += f"| {'N/A':>{col_w}} "
            print(row)

    print(f"\nAccuracy")
    header = f"{'Model':<12}"
    for df in data_folds:
        header += f"| {df:>{col_w}} "
    print(header)
    print("-" * (12 + (col_w + 3) * len(data_folds)))
    for mf in model_folds:
        row = f"{mf:<12}"
        for df in data_folds:
            k = (mf, df)
            if k in results:
                row += f"| {results[k]['accuracy']:>{col_w}.4f} "
            else:
                row += f"| {'N/A':>{col_w}} "
        print(row)

    # --- HT Mixed Detection Evaluation Metrics ---
    print(f"\nEquivalent HT Detection from Fill (Pos: equivHT, Neg: Fill)")
    header = f"{'Model':<12}| {'Detection':>{col_w}} | {'FPR':>{col_w}} | {'Accuracy':>{col_w}} "
    print(header)
    print("-" * (12 + (col_w + 3) * 3))
    for mf in model_folds:
        row = f"{mf:<12}"
        if mf in ht_results and ht_results[mf] is not None:
            dr = ht_results[mf]['detection_rate']
            fpr = ht_results[mf]['fpr']
            acc = ht_results[mf]['accuracy']
            row += f"| {dr:>{col_w}.4f} | {fpr:>{col_w}.4f} | {acc:>{col_w}.4f} "
        else:
            row += f"| {'N/A':>{col_w}} | {'N/A':>{col_w}} | {'N/A':>{col_w}} "
        print(row)
    print()

# -------------- Entry Point --------------

def main():
    parser = argparse.ArgumentParser(description="Offline data cross-evaluation program")
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='Model weights root directory (Default: PROJECT_ROOT/weights/pre)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Offline data directory (Default: PROJECT_ROOT/data/offline_data)')
    parser.add_argument('--model_subdir', type=str, default='siam128_333',
                        help='Model subdirectory name (default: siam128_333)')
    parser.add_argument('--model_filename', type=str, default='snet_bestLoss.pth',
                        help='Model filename (default: snet_bestLoss.pth)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--time_length', type=int, default=1000)
    parser.add_argument('--num_ref_points', type=int, default=1,
                        help='Number of reference points; evaluation is done independently for each before voting (default: 1)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    weights_dir = args.weights_dir or os.path.join(PROJECT_ROOT, 'weights', 'pre')
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, 'data', 'offline_data')

    model_folds = discover_folds_model(weights_dir, args.model_subdir, args.model_filename)
    data_folds = discover_folds_data(data_dir)

    if not model_folds:
        print(f"[ERROR] No fold models found under {weights_dir}")
        return
    if not data_folds:
        print(f"[ERROR] No fold data found under {data_dir}")
        return

    model_fold_names = [name for name, _ in model_folds]
    print(f"Model dir   : {weights_dir}")
    print(f"Data dir    : {data_dir}")
    print(f"Model folds : {model_fold_names}")
    print(f"Data folds  : {data_folds}")
    print(f"Ref points  : {args.num_ref_points}")

    # Pre-load HT data for fold1 (shared between both models)
    fold1_class_data, fold1_unknown_data = load_fold_data(data_dir, 'fold1')
    if fold1_unknown_data is None:
        print(f"[WARN] No class_1_unknown data for fold1, HT detection rate will be skipped")

    results = {}        # (model_fold, data_fold) → metrics
    ht_results = {}     # model_fold → ht_detection_rate

    for model_fold, model_path in model_folds:
        print(f"\nLoading model: {model_fold} from {model_path}")
        sNet = SiameseNet(
            HSE(), CTE_3D(args.time_length), patch_size=args.patch_size)
        sNet.load_state_dict(torch.load(model_path, map_location=args.device))
        sNet.to(args.device)
        sNet.eval()

        # 1. Standard metrics: Evaluate on train+val data for each fold
        for data_fold in data_folds:
            print(f"  {model_fold} → {data_fold} ...", end="", flush=True)
            if data_fold == 'fold1' and fold1_class_data is not None:
                class_data = fold1_class_data
            else:
                class_data, _ = load_fold_data(data_dir, data_fold)

            if len(class_data) < 2:
                print(" skip (incomplete data)")
                continue

            metrics, _ = evaluate_fold(
                sNet, class_data, None,
                args.batch_size, args.device, args.num_ref_points)
            if metrics is not None:
                results[(model_fold, data_fold)] = metrics
                print(f" Acc={metrics['accuracy']:.4f}")
            else:
                print(" skip")
                
        # 2. HT detection rate: Mixed evaluation of fold1 class_1_unknown and class_2
        if fold1_unknown_data is not None:
            print(f"  {model_fold} → fold1 HT ...", end="", flush=True)
            _, ht_metrics_dict = evaluate_fold(
                sNet, fold1_class_data, fold1_unknown_data,
                args.batch_size, args.device, args.num_ref_points)
            
            if ht_metrics_dict is not None:
                ht_results[model_fold] = ht_metrics_dict
                print(f" DR={ht_metrics_dict['detection_rate']:.4f}, FPR={ht_metrics_dict['fpr']:.4f}")
            else:
                print(" skip")

        del sNet
        torch.cuda.empty_cache()

    print_results(results, ht_results, model_fold_names, data_folds)


if __name__ == '__main__':
    main()