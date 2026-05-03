# -*- coding: utf-8 -*-
"""
Siamese Net Training Program

Data Mode:
    - online  : Real-time sampling of pixel pairs from raw .npz/.npy data.
               Automatically splits space into fold1/fold2; each fold is trained independently.
               equivHT is extracted within each fold and merged into class2; train/val are randomly partitioned at the pixel level.
    - offline : Loaded from pre-generated offline .npy data (Privacy protection, no raw data required).

Usage:
    # Online mode - train all folds
    python main.py --data_mode online

    # Online mode - specify a fold
    python main.py --data_mode online --fold fold1

    # Offline mode - fold1
    python main.py --data_mode offline --fold fold1
    python main.py --data_mode offline --fold fold2

    # Common parameters
    python main.py --data_mode online \\
        --patch_size 128 --neighborhood_size 3 --time_length 1000 \\
        --batch_size 8 --epochs 100 --lr 0.0001

        
Offline Data Generation:
    See offline_data_generator.py; splits raw data into fold1/fold2 spatially.
    Each fold is sampled independently, with train/val split randomly by ratio.
    class_1_unknown (equivalent Trojan pixels) is extracted from class1 and merged into class2 during training.

Online Data Preparation:
    See online_data_prepare.py; provides fold splitting, equivHT extraction, 
    and coordinate pre-partitioning for online training. equivHT coordinates are saved to data/online_data/equiv_ht/{fold}/.

Evaluation:
    See evaluate.py; batch inference evaluation program based on offline data.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import random
import time
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from model import CTE_3D, HSE, SiameseNet

# Siamese Network pairing labels: 0=Different classes pair, 1=Same class pair
PAIR_DIFF = 0
PAIR_SAME = 1


# -------------- Online Dataset --------------


# Randomly apply rotation and flipping
def transform(matrix, angle=None, doflip=False, flip1=False):
    if angle is not None:
        # Randomly rotate the matrix
        if angle == 90:
            matrix = np.rot90(matrix)
        elif angle == 180:
            matrix = np.rot90(matrix, k=2)
        elif angle == 270:
            matrix = np.rot90(matrix, k=3)
    if doflip:
        # Flip the matrix (horizontal or vertical)
        if flip1:
            matrix = np.flip(matrix, 1)  # Horizontal flip
        else:
            matrix = np.flip(matrix, 0)  # Vertical flip

    return matrix


def transform_3d(matrix_3d, angle=None, doflip=False, flip1=False):
    transformed = []
    for i in range(matrix_3d.shape[0]):
        # Process each slice independently
        transformed_slice = transform(matrix_3d[i], angle, doflip, flip1)
        transformed.append(transformed_slice)

    # Standardize all slice dimensions (Important!)
    try:
        return np.stack(transformed, axis=0)
    except ValueError as e:
        print(f"Shape incompatibility error: {e}")
        return matrix_3d  # Return raw data or handle exception


class ImgTSNetDataset(Dataset):
    """
    Online Dataset: Real-time sampling of pixel pairs from preprocessed fold data.

    Data provided by online_data_prepare.prepare_fold_data(), containing:
    - data_np, normal_data_np, label_np: Full data for the fold
    - train_coords / val_coords: Pre-partitioned coordinate pools for class1/class2 (pixel-level non-overlapping)
    - equiv_ht_coords: equivHT coordinates (excluded from class1, merged into class2)

    During training, the model only sees two classes: Logic (class1) vs. Fill (class2 + equivHT).
    """
    def __init__(self, data_np, normal_data_np, label_np,
                 phase_coords, equiv_ht_coords,
                 samples_num, neighborhood_size, batch_size, patch_size):
        """
        Parameters:
            data_np: Raw time-series data (T, H, W)
            normal_data_np: Normalized time-series data (T, H, W)
            label_np: Labels (H, W)
            phase_coords: {'class1': (rows, cols), 'class2': (rows, cols)}
            equiv_ht_coords: (N, 2) equivHT coordinates
            samples_num: Number of samples per epoch
            neighborhood_size: Neighborhood size
            batch_size: Batch size
            patch_size: Spatial patch size
        """
        self.data = data_np
        self.normal_data_np = normal_data_np
        self.label = label_np
        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.samples_num = samples_num
        self.directions = self.generate_offsets(self.neighborhood_size)

        # class1 coordinate pool (equivHT excluded)
        self.class1_rows, self.class1_cols = phase_coords['class1']
        self.n_class1 = len(self.class1_rows)

        # Merge class2 coordinate pool + equivHT
        cls2_rows, cls2_cols = phase_coords['class2']
        if len(equiv_ht_coords) > 0:
            ht_rows = equiv_ht_coords[:, 0]
            ht_cols = equiv_ht_coords[:, 1]
            self.class2_rows = np.concatenate([cls2_rows, ht_rows], axis=0)
            self.class2_cols = np.concatenate([cls2_cols, ht_cols], axis=0)
        else:
            self.class2_rows = cls2_rows
            self.class2_cols = cls2_cols
        self.n_class2 = len(self.class2_rows)

        print(f"  [Online Dataset] class1={self.n_class1}, class2={self.n_class2} "
              f"(including {len(equiv_ht_coords)} equivHT), "
              f"data shape={data_np.shape}")

    def generate_offsets(self, n: int):
        """
        Generate coordinate offset list for an n x n grid relative to center (0,0).
        n must be odd, e.g., 3, 5, 7 ...

        Returns: List[(dy, dx)], arranged in row-major order (top-to-bottom, left-to-right).
        """
        if n % 2 == 0 or n < 1:
            raise ValueError("n must be a positive odd integer")
        half = n // 2
        offsets = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                offsets.append((dy, dx))
        return offsets

    def __len__(self):
        """Returns dataset size"""
        return self.samples_num

    def _sample_points(self, pair_label, height, width):
        """Sample two points from the pre-partitioned coordinate pool based on the pairing label: PAIR_SAME=Same class, PAIR_DIFF=Different classes"""
        if pair_label == PAIR_DIFF:
            i1 = random.randint(0, self.n_class1 - 1)
            idx1 = (self.class1_rows[i1], self.class1_cols[i1])
            i2 = random.randint(0, self.n_class2 - 1)
            idx2 = (self.class2_rows[i2], self.class2_cols[i2])
        else:  # PAIR_SAME
            # Same class pairing: randomly select class1 or class2
            if random.random() < 0.5:
                i1 = random.randint(0, self.n_class1 - 1)
                idx1 = (self.class1_rows[i1], self.class1_cols[i1])
                i2 = random.randint(0, self.n_class1 - 1)
                idx2 = (self.class1_rows[i2], self.class1_cols[i2])
            else:
                i1 = random.randint(0, self.n_class2 - 1)
                idx1 = (self.class2_rows[i1], self.class2_cols[i1])
                i2 = random.randint(0, self.n_class2 - 1)
                idx2 = (self.class2_rows[i2], self.class2_cols[i2])
        return idx1, idx2

    def _extract_sequence(self, img3d, center):
        """Extract time series from 3D image based on offsets"""
        # img1Ddata_Line(img3d[:, 1, 1])
        grid = np.array([(center[0] + dx, center[1] + dy) for dx, dy in self.directions])
        temp = img3d[:, grid[:, 0], grid[:, 1]]  # shape (T, K*K)
        grid_sequence = temp.reshape(-1, self.neighborhood_size, self.neighborhood_size)
        grid_sequence = grid_sequence.transpose(1, 2, 0)
        return grid_sequence

    def __getitem__(self, idx):
        """Returns data and labels"""
        height, width = self.label.shape

        label = PAIR_SAME if random.randint(1, 10) > 5 else PAIR_DIFF

        idx1, idx2 = self._sample_points(label, height, width)

        seq1 = self._extract_sequence(self.data, idx1)
        seq2 = self._extract_sequence(self.data, idx2)
        data_seq = np.stack([seq1, seq2], axis=0)

        # Randomly select a frame to crop the spatial patch
        frame_index = np.random.randint(0, self.normal_data_np.shape[0])
        frame = self.normal_data_np[frame_index]
        frame_ = np.tile(frame, (3, 3))
        patch_size = self.patch_size
        half_patch = patch_size // 2

        def _extract_patch_from_frame(center, frame, frame_):
            cx = center[0] + frame.shape[0]
            cy = center[1] + frame.shape[1]
            sx, ex = cx - half_patch, cx + half_patch + (patch_size % 2)
            sy, ey = cy - half_patch, cy + half_patch + (patch_size % 2)
            return frame_[sx:ex, sy:ey]

        patch1 = _extract_patch_from_frame(idx1, frame, frame_)
        patch2 = _extract_patch_from_frame(idx2, frame, frame_)
        data_img = np.stack([patch1, patch2], axis=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = {
            'imgs': torch.tensor(data_img).float().to(device),
            'seqs': torch.tensor(data_seq).float().to(device),
        }

        return data, label


def get_dataloader(data_np, normal_data_np, label_np,
                   phase_coords, equiv_ht_coords,
                   batches_num, neighborhood_size, batch_size, patch_size,
                   shuffle=True, num_workers=0):
    dataset = ImgTSNetDataset(
        data_np, normal_data_np, label_np,
        phase_coords, equiv_ht_coords,
        batches_num * batch_size, neighborhood_size, batch_size, patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# -------------- Offline Dataset --------------

class OfflineImgTSNetDataset(Dataset):
    """
    Offline Dataset: Loads pre-sampled (seq, img) data from pre-generated .npy files to construct Siamese network pairs.

    class_1_unknown (equivalent Trojan pixels) is merged into class_2 pool during training, and labels are treated as Fill.
    The model does not distinguish between class_2 and class_1_unknown during training; it only sees Logic vs. Fill.

    Directory Structure:
        {data_dir}/{fold}/train|val/
        ├---- class_1_seqs.npy           Logic pixels
        ├---- class_1_imgs.npy
        ├---- class_2_seqs.npy           Fill pixels
        ├---- class_2_imgs.npy
        ├---- class_1_unknown_seqs.npy   equivalent Trojans (merged into class_2)
        └---- class_1_unknown_imgs.npy
    """

    def __init__(self, data_dir, fold, phase, samples_num):
        self.phase = phase
        self.samples_num = samples_num

        fold_phase_dir = os.path.join(data_dir, fold, phase)

        # class_1: Logic pixels
        self.class_data = {}
        self.class_data[1] = {
            'seqs': np.load(os.path.join(fold_phase_dir, 'class_1_seqs.npy')),
            'imgs': np.load(os.path.join(fold_phase_dir, 'class_1_imgs.npy')),
        }
        print(f"  [Offline] class 1: {self.class_data[1]['seqs'].shape[0]} samples loaded")

        # class_2: Fill pixels + class_1_unknown merged (Features from Logic, but label treated as Fill)
        cls2_seqs = np.load(os.path.join(fold_phase_dir, 'class_2_seqs.npy'))
        cls2_imgs = np.load(os.path.join(fold_phase_dir, 'class_2_imgs.npy'))

        unknown_seq_path = os.path.join(fold_phase_dir, 'class_1_unknown_seqs.npy')
        unknown_img_path = os.path.join(fold_phase_dir, 'class_1_unknown_imgs.npy')
        if os.path.isfile(unknown_seq_path) and os.path.isfile(unknown_img_path):
            unk_seqs = np.load(unknown_seq_path)
            unk_imgs = np.load(unknown_img_path)
            cls2_seqs = np.concatenate([cls2_seqs, unk_seqs], axis=0)
            cls2_imgs = np.concatenate([cls2_imgs, unk_imgs], axis=0)
            print(f"  [Offline] class 1_unknown: {unk_seqs.shape[0]} samples merged into class 2")

        self.class_data[2] = {
            'seqs': cls2_seqs,
            'imgs': cls2_imgs,
        }
        print(f"  [Offline] class 2 (with unknown): {cls2_seqs.shape[0]} samples loaded")

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        # Randomly determine the pairing type
        label = PAIR_SAME if random.randint(1, 10) > 5 else PAIR_DIFF

        if label == PAIR_SAME:
            # Same class pairing: two points from the same class
            cls = random.choice([1, 2])
            n = self.class_data[cls]['seqs'].shape[0]
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            seq1 = self.class_data[cls]['seqs'][i1]
            img1 = self.class_data[cls]['imgs'][i1]
            seq2 = self.class_data[cls]['seqs'][i2]
            img2 = self.class_data[cls]['imgs'][i2]
        else:
            # Different classes pairing: two points from different classes
            i1 = random.randint(0, self.class_data[1]['seqs'].shape[0] - 1)
            i2 = random.randint(0, self.class_data[2]['seqs'].shape[0] - 1)
            seq1 = self.class_data[1]['seqs'][i1]
            img1 = self.class_data[1]['imgs'][i1]
            seq2 = self.class_data[2]['seqs'][i2]
            img2 = self.class_data[2]['imgs'][i2]

        # seq shape: (neighborhood_size, neighborhood_size, T) → Needs transpose to (T, N, N) before stack
        # Consistent with ImgTSNetDataset._extract_sequence output format: (N, N, T)
        data_seq = np.stack([seq1, seq2], axis=0)   # (2, N, N, T)
        data_img = np.stack([img1, img2], axis=0)   # (2, patch_size, patch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = {
            'imgs': torch.tensor(data_img).float().to(device),
            'seqs': torch.tensor(data_seq).float().to(device),
        }

        return data, label


def get_offline_dataloader(data_dir, fold, phase, batches_num, batch_size,
                           shuffle=True, num_workers=0):
    dataset = OfflineImgTSNetDataset(data_dir, fold, phase, batches_num * batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive Loss: Pairs of the same class should be close, while pairs of different classes should be far apart (no penalty beyond margin).
    label=1 (PAIR_SAME) Same class → Minimize distance; label=0 (PAIR_DIFF) Different classes → Push apart to margin.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +(1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
class Network:
    """
    Training Manager: Encapsulates the model, optimizer, training loop, and evaluation logic.

    Training Strategy:
        - Initially freeze encoder parameters, training only the fusion layers of SiameseNet.
        - Unfreeze all encoder and tnet parameters when reaching the first half of the early stopping patience.
        - Use CosineAnnealingWarmRestarts for learning rate scheduling.
        - Joint Loss: 0.5×CE + 0.2×contrastive_ts + 0.1×(contrastive_conv1~4).

    Saved Model Files:
        - snet_bestLoss.pth : Updated when validation loss reaches a new low (state_dict).
        - best_by_accuracy.pth : Saved when validation accuracy reaches a new high (contains metadata like epoch).
        - snet_final.pth : Saves the state_dict of the best model when training completes.
    """
    def __init__(self, model, train_dataloader, val_dataloader, output_dir,
                 loss_function, learning_rate, device, num_classes,
                 neighborhood_size, patch_size):
        print(device)
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function.to(device)
        self.contrastive_loss = ContrastiveLoss().to(device)
        self.loss_lambda=0.5
        self.neighborhood_size = neighborhood_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_batches_num = len(train_dataloader)
        self.val_batches_num = len(val_dataloader)
        self.output_dir = output_dir
        # best metrics storage for accuracy & AUC
        self.best_metrix = {
            'best_acc': -float('inf'),
            'best_auc': -float('inf')
        }
        # Partially freeze model parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-9
        )

        # Log and model output directories
        log_dir = os.path.join(output_dir, 'log')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.early_stopping_patience = 20  # Adjustable wait epoch count
        self.early_stopping_counter = 0
        self.early_stopping_best_loss = float("inf")

    def train(self, epoch_num):
        print("train phase...")

        best_model = self.model
        best_loss = float('inf')

        # Epoch loop
        # Every epoch
        for epoch in range(epoch_num):

            torch.cuda.empty_cache()
            running_loss, running_acc = 0, 0

            # validation & early stopping
            eval_result = self.evaluate(epoch, matrix_save_flag=True)

            torch.cuda.empty_cache()
            val_loss = eval_result['loss']
            if val_loss < self.early_stopping_best_loss:
                self.early_stopping_best_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"EarlyStopping counter: {self.early_stopping_counter} out of {self.early_stopping_patience}")

            if self.early_stopping_counter >= self.early_stopping_patience//2:
                # Unfreeze encoder and tnet, allowing pre-trained sub-modules to participate in fine-tuning
                print("Requires_grad = True")
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.tnet.parameters():
                    param.requires_grad = True

            if self.early_stopping_counter >= self.early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                break

            self.model.train()
            for batch_idx, (data_batch, label_batch) in enumerate(self.train_dataloader):

                label_batch = label_batch.to(self.device)

                output,(left_ts_features,right_ts_features,left_img_features,right_img_features) = self.model(data_batch)

                loss1 = self.loss_function(output, label_batch)
                gap = torch.nn.AdaptiveAvgPool2d(1)


                # Contrastive Loss
                loss_contrast_ts = self.contrastive_loss(left_ts_features, right_ts_features, label_batch)
                
                left = gap(left_img_features['conv1'])    # [batch, 3, 1, 1]
                right = gap(right_img_features['conv1'])  # [batch, 3, 1, 1]
                left = left.view(left.size(0), -1)        # [batch, 3]
                right = right.view(right.size(0), -1)     # [batch, 3]
                loss_contrast_patch1 = self.contrastive_loss(left,right , label_batch)

                left = gap(left_img_features['conv2'])
                right = gap(right_img_features['conv2'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch2 = self.contrastive_loss(left,right , label_batch)
                
                left = gap(left_img_features['conv3'])
                right = gap(right_img_features['conv3'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch3 = self.contrastive_loss(left,right , label_batch)
                
                
                left = gap(left_img_features['conv4'])
                right = gap(right_img_features['conv4'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch4 = self.contrastive_loss(left,right , label_batch)
                
                del left,right
                total_loss = 0.5*loss1 + 0.2*loss_contrast_ts+0.1*loss_contrast_patch1+0.1*loss_contrast_patch2+0.1*loss_contrast_patch3+0.1*loss_contrast_patch4
                
                self.optimizer.zero_grad()  # Clear gradients first
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.data.cpu().numpy()

                accuracy = calculate_accuracy(output, label_batch)

                running_acc += accuracy

                current_lr = self.optimizer.param_groups[0]['lr']
                progress = f"Epoch {epoch}, Batch {batch_idx:5d}/{self.train_batches_num:5d}, Loss: {total_loss.item():.10f}, Accuracy: {accuracy:.10f}, LR: {current_lr:.10f}"
                print(f'\r{progress}', end='', flush=True)

            # Record Loss
            epoch_loss = running_loss / self.train_batches_num
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)

            epoch_accuracy = running_acc / self.train_batches_num

            # Get the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
            progress = f"Epoch {epoch}, Batch {self.train_batches_num:5d}/{self.train_batches_num:5d}, Loss: {epoch_loss:.10f}, Accuracy: {epoch_accuracy:.10f}, Time: {current_time}"
            print(f'\r{progress}', end='\n')

            if epoch_loss < best_loss:
                print('update best model......')
                best_model = copy.deepcopy(self.model)
                best_loss = epoch_loss
                save_path = os.path.join(self.output_dir, 'snet_bestLoss.pth')
                torch.save(best_model.state_dict(), save_path)

            self.scheduler.step()  # Update learning rate


        save_path = os.path.join(self.output_dir, 'snet_final.pth')
        print(f"Saving model into {save_path}...")
        torch.save(best_model.state_dict(), save_path)
        print("final model saved..\n")
        self.writer.close()

    def evaluate(self, epoch, matrix_save_flag=False):
        print("evaluate phase...")

        # Evaluation mode setting
        self.model.eval()

        all_confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        torch.cuda.empty_cache()
        total_loss = 0.0
        total_acc = 0
        total_auc = 0
        valid_auc_count = 0
        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, (data_batch, label_batch) in enumerate(self.val_dataloader):

                label_batch = label_batch.to(self.device)

                output,(left_ts_features,right_ts_features,left_img_features,right_img_features) = self.model(data_batch)

                loss1 = self.loss_function(output, label_batch)
                gap = torch.nn.AdaptiveAvgPool2d(1)


                # Contrastive Loss
                loss_contrast_ts = self.contrastive_loss(left_ts_features, right_ts_features, label_batch)
                
                left = gap(left_img_features['conv1'])    # [batch, 3, 1, 1]
                right = gap(right_img_features['conv1'])  # [batch, 3, 1, 1]
                left = left.view(left.size(0), -1)        # [batch, 3]
                right = right.view(right.size(0), -1)     # [batch, 3]
                loss_contrast_patch1 = self.contrastive_loss(left,right , label_batch)

                left = gap(left_img_features['conv2'])
                right = gap(right_img_features['conv2'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch2 = self.contrastive_loss(left,right , label_batch)
                
                left = gap(left_img_features['conv3'])
                right = gap(right_img_features['conv3'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch3 = self.contrastive_loss(left,right , label_batch)
                
                
                left = gap(left_img_features['conv4'])
                right = gap(right_img_features['conv4'])
                left = left.view(left.size(0), -1)
                right = right.view(right.size(0), -1)
                loss_contrast_patch4 = self.contrastive_loss(left,right , label_batch)
                
                del left,right
                loss = 0.5*loss1 + 0.2*loss_contrast_ts+0.1*loss_contrast_patch1+0.1*loss_contrast_patch2+0.1*loss_contrast_patch3+0.1*loss_contrast_patch4
                

                # Confusion matrix accumulation
                if output.dim() == label_batch.dim() + 1:  # (B, num_classes, ...)
                    preds = torch.argmax(output, dim=1)
                else:
                    preds = output
                confusion = calculate_confusion_matrix(preds, label_batch, self.num_classes)
                all_confusion += confusion

                total_loss += loss.data.cpu().numpy()
                total_acc += calculate_accuracy(output, label_batch)

        avg_loss = total_loss / self.val_batches_num
        avg_acc = total_acc / self.val_batches_num
        # avg_auc = total_auc / self.val_batches_num

        print(f"\nValidation metrics - Loss: {avg_loss:.10f}, Accuracy: {avg_acc:.10f}")
        print(all_confusion)
        # Calculate all metrics
        metrics = self.calculate_metrics(all_confusion)

        # Print full evaluation results
        print(f"Evaluation Results (val):")
        print(f"Average Loss: {total_loss / (self.val_batches_num)}")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
        print(f"Kappa Coefficient: {metrics['kappa']:.4f}")

        pair_names = ['Diff(Different classes)', 'Same(Same class)']
        for i in range(self.num_classes):
            print(f"\n{pair_names[i]}:")
            print(f"  Precision: {metrics['precision'][i]:.4f}")
            print(f"  Recall:    {metrics['recall'][i]:.4f}")
            print(f"  F1-score:  {metrics['f1'][i]:.4f}")
        print('\n')
        # Record metrics
        if matrix_save_flag:

            # Global step can also use self.global_step, or custom such as current timestamp or epoch
            self.writer.add_scalar("Loss/val", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/val", avg_acc, epoch)
            # self.writer.add_scalar("AUC/val", avg_auc, epoch)

            # # 1. Save confusion matrix as npy
            # np.save(f'confusion_matrix_epoch_{epoch}.npy', all_confusion)

            # 2. Visualize and write to TensorBoard
            # fig = plot_confusion_matrix(all_confusion)
            # self.writer.add_figure('Val/Confusion_Matrix', fig, epoch)
            # plt.close(fig)

            # 3. Can also save as text (optional)
            self.writer.add_text('Val/Confusion_Matrix', np.array2string(all_confusion), epoch)

            # save best based on accuracy
            if avg_acc > self.best_metrix['best_acc']:
                self.best_metrix['best_acc'] = avg_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_metric_value': avg_acc,
                    'metric_name': 'accuracy',
                }, os.path.join(self.output_dir, 'best_by_accuracy.pth'))
            
            
            self.model.train()

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            
        }

    def calculate_metrics(self, confusion_matrix):
        """Calculate multi-class evaluation metrics"""
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': 0,
            'macro_f1': 0,
            'kappa': 0
        }

        # Calculate base metrics
        total = confusion_matrix.sum()
        diag_sum = np.diag(confusion_matrix).sum()
        metrics['accuracy'] = diag_sum / total

        # Calculate metrics for each category
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            tn = total - (tp + fp + fn)

            # Precision/Recall/F1
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)

        metrics['macro_f1'] = np.mean(metrics['f1'])

        # Kappa coefficient
        po = diag_sum / total
        pe = (confusion_matrix.sum(0) @ confusion_matrix.sum(1)) / (total ** 2)
        metrics['kappa'] = (po - pe) / (1 - pe + 1e-10)

        return metrics


def calculate_confusion_matrix(preds, labels, num_classes):
    # Ensure they are numpy arrays and on CPU
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    mask = (labels >= 0) & (labels < num_classes)
    confusion, _, _ = np.histogram2d(labels[mask], preds[mask],
                                     bins=(num_classes, num_classes),
                                     range=[[0, num_classes], [0, num_classes]])
    return confusion.astype(int)


def plot_confusion_matrix(confusion_matrix, class_names=None):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(np.arange(confusion_matrix.shape[0]))
        ax.set_yticks(np.arange(confusion_matrix.shape[1]))
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    return fig


def calculate_accuracy(output, labels):
    """
    Calculate accuracy, applicable to classification tasks.

    Parameters:
    - output: Prediction result, shape (batch_size, num_classes)
    - labels: True labels, shape (batch_size, )

    Returns:
    - accuracy: Accuracy (correct samples / total samples)
    """
    # Get the predicted class for each sample
    preds = torch.argmax(output, dim=1)
    # Count correctly predicted samples
    correct = (preds == labels).sum().item()
    # Count total samples
    total = labels.size(0)
    # Calculate accuracy
    accuracy = correct / total
    return accuracy


def calculate_auc_roc(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate AUC-ROC score, applicable to binary or multi-class tasks.

    Parameters:
    outputs (torch.Tensor): Model output, shape (batch_size, num_classes) or (batch_size,).
    labels (torch.Tensor): True labels, shape (batch_size,).

    Returns:
    float: Mean AUC-ROC score across all classes.
    """
    if outputs.ndim == 1 or outputs.shape[1] == 1:
        # Binary classification case, use sigmoid to convert logits to probabilities
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        return roc_auc_score(labels_np, probs)
    else:
        # Multi-class case, use softmax to convert logits to probabilities
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        
        # Convert labels to one-hot encoding
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[1]).cpu().numpy()
        # Calculate macro-average AUC-ROC using one-vs-rest approach
        return roc_auc_score(labels_onehot, probs, average="macro", multi_class="ovr")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Siamese Net Training Programmage")
    # ---- Data Mode ----
    parser.add_argument('--data_mode', type=str, default='online', choices=['online', 'offline'],
                        help='Data mode: online=sample online from raw data, offline=use pre-generated offline data (default: online)')
    parser.add_argument('--fold', type=str, default=None, choices=['fold1', 'fold2'],
                        help='Specify fold; if not specified, train all folds (default: all)')
    parser.add_argument('--offline_data_dir', type=str, default=None,
                        help='Root directory for offline data, default data/offline_data (offline mode only)')
    parser.add_argument('--num_equiv_ht', type=int, default=50,
                        help='Number of equivHT per fold in online mode (default: 50)')
    # ---- Model Architecture Parameters ----
    parser.add_argument('--time_length', type=int, default=1000,
                        help='Temporal length T, i.e., time dimension for 3D CNN input (default: 1000)')
    parser.add_argument('--neighborhood_size', type=int, default=3,
                        help='Neighborhood size N, extracts N×N×T time series (must be odd, default: 3)')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Spatial patch size, UNet encoder input patch_size×patch_size (default: 128)')
    # ---- Training Hyperparameters ----
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--train_batches', type=int, default=100,
                        help='Number of training batches per epoch (default: 100)')
    parser.add_argument('--val_batches', type=int, default=50,
                        help='Number of validation batches per epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    # ---- Others ----
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device: cuda or cpu (default: cuda)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    online_data_dir = os.path.join(data_dir, 'online_data')
    weights_dir = os.path.join(PROJECT_ROOT, 'weights')
    pre_dir = os.path.join(weights_dir, 'pre')
    test_train_dir = os.path.join(weights_dir, 'test_train')

    # Model output subdirectory name
    model_subdir = f'siam{args.patch_size}_{args.neighborhood_size}{args.neighborhood_size}{args.neighborhood_size}'

    num_classes = 2

    # Determine the list of folds to train
    folds = [args.fold] if args.fold else ['fold1', 'fold2']

    print(f'PROJECT_ROOT  : {PROJECT_ROOT}')
    print(f'Data mode     : {args.data_mode}')
    print(f'Folds         : {folds}')

    for fold in folds:
        print(f"\n{'#'*60}")
        print(f"Training fold: {fold}")
        print(f"{'#'*60}")

        # ------------ Construct Data Loaders ------------
        if args.data_mode == 'online':
            from online_data_prepare import prepare_fold_data, save_coords

            data_np, normal_data_np, label_np, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug = \
                prepare_fold_data(
                    online_data_dir, fold,
                    val_ratio=0.2,
                    num_equiv_ht=args.num_equiv_ht,
                    time_length=args.time_length,
                    seed=args.seed,
                )

            save_coords(online_data_dir, fold, equiv_ht_coords, train_coords, val_coords, equiv_ht_coords_pre_aug)

            train_dataloader = get_dataloader(
                data_np, normal_data_np, label_np,
                train_coords, equiv_ht_coords,
                batches_num=args.train_batches,
                neighborhood_size=args.neighborhood_size,
                batch_size=args.batch_size, patch_size=args.patch_size)
            val_dataloader = get_dataloader(
                data_np, normal_data_np, label_np,
                val_coords, equiv_ht_coords,
                batches_num=args.val_batches,
                neighborhood_size=args.neighborhood_size,
                batch_size=args.batch_size, patch_size=args.patch_size)
            output_dir = os.path.join(pre_dir, fold, model_subdir)

        else:  # offline
            offline_data_dir = args.offline_data_dir or os.path.join(data_dir, 'offline_data')
            train_dataloader = get_offline_dataloader(
                offline_data_dir, fold=fold, phase='train',
                batches_num=args.train_batches, batch_size=args.batch_size)
            val_dataloader = get_offline_dataloader(
                offline_data_dir, fold=fold, phase='val',
                batches_num=args.val_batches, batch_size=args.batch_size)
            output_dir = os.path.join(test_train_dir, fold, model_subdir)

        os.makedirs(output_dir, exist_ok=True)
        print(f'Output dir    : {output_dir}')

        # ------------ Construct Model ------------
        cte_3d = CTE_3D(args.time_length)
        hse = HSE()
        sNet = SiameseNet(hse, cte_3d, patch_size=args.patch_size)

        # Load pre-trained full model
        # pretrained_path = os.path.join(pre_dir, 'snet_pretrained.pth')
        # if os.path.isfile(pretrained_path):
        #     print(f"Loading pretrained model from {pretrained_path}...")
        #     sNet.load_state_dict(torch.load(pretrained_path, map_location=args.device))

        loss_function = torch.nn.CrossEntropyLoss()

        network = Network(
            model=sNet,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=output_dir,
            loss_function=loss_function,
            learning_rate=args.lr,
            device=args.device,
            num_classes=num_classes,
            neighborhood_size=args.neighborhood_size,
            patch_size=args.patch_size,
        )
        network.train(epoch_num=args.epochs)

        del sNet, network
        torch.cuda.empty_cache()