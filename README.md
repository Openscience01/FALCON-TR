# Multi-Scale Features Comparison of Thermal Radiation for Hardware Trojan Detection in Fill Regions.


This repository contains the source code and pre-trained models for the framework described in the paper.
Artifact for FALCON-TR: Fine-grained and Regional-level Features Comparison through a Siamese Network for TR-based HT Detection

## Overview

The framework uses a Siamese network architecture to compare multi-scale thermal radiation features. It comprises two main components:
*   **3D-CTE**: A 3D Convolutional Temporal Encoder for time-series analysis.
*   **HSE**: A Hierarchical Spatial Encoder for spatial feature extraction.

We utilize a Fold-Rotating Detection Strategy, where models are independently trained and inferred on partitioned spatial folds, and their individual results are aggregated to reliably generate the final detection results.

## Workflow

### 1. Model Training

**With original data (online mode):**

```bash
# fold1: includes equivalent HT (equivalent trojan pixels)
python main.py --data_mode online --fold fold1 --num_equiv_ht 3000

# fold2: without equivalent HT
python main.py --data_mode online --fold fold2 --num_equiv_ht 0
```

Models are saved to `weights/pre/{fold}/siam128_333/`.  (Pre-executed)

****With offline data (offline mode, Verify training pipeline):**

```bash
python main.py --data_mode offline --fold fold1
python main.py --data_mode offline --fold fold2
```

Models are saved to `weights/test_train/{fold}/siam128_333/`.

### 2. Generate Offline Data (Pre-executed)

Requires equivHT coordinates from Step 1 (online training):

```bash
 python .\offline_data_generator.py \
    --samples_per_class 500 \
    --equiv_ht_dir data\online_data\equiv_ht \
    --val_ratio 0.2 \
    --mask data\online_data\mask.npy \
    --data data\online_data\data.npz \
    --num_equiv_ht 100
```

Output structure:

```
data/offline_data/
├── fold1/
│   ├── train/  (class_1, class_2, class_1_unknown)
│   └── val/    (class_1, class_2, class_1_unknown)
└── fold2/
    ├── train/  (class_1, class_2)
    └── val/    (class_1, class_2)
```

### 3. Evaluation

```bash
python evaluate.py --batch_size 32 \
    --data_dir data\offline_data \
    --weights_dir weights\pre \
    --num_ref_points 11 

```

This produces:
- **Classification metrics**: Precision/Recall/Accuracy for each (model_fold, data_fold) pair
- **HT Evaluation**: Evaluates each fold's model on a mixed dataset of fold1's class_1_unknown (treated as positive/HT) and normal class_2 data (treated as negative). 

Example Output:

```
Model dir   : weights\pre
Data dir    : data\offline_data
Model folds : ['fold1', 'fold2']
Data folds  : ['fold1', 'fold2']
Ref points  : 11

Loading model: fold1 from weights\pre\fold1\siam128_333\snet_bestLoss.pth
  fold1 → fold1 ... Acc=0.9560
  fold1 → fold2 ... Acc=0.9520
  fold1 → fold1 HT ... DR=0.9464, FPR=0.0000

Loading model: fold2 from weights\pre\fold2\siam128_333\snet_bestLoss.pth
  fold2 → fold1 ... Acc=0.9750
  fold2 → fold2 ... Acc=0.9800
  fold2 → fold1 HT ... DR=0.9821, FPR=0.0000

Logic Precision
Model       |        fold1 |        fold2 
------------------------------------------
fold1       |       1.0000 |       1.0000 
fold2       |       1.0000 |       1.0000 

[ ... Logic Recall, Fill Precision/Recall, and Accuracy tables omitted for brevity ... ]

Equivalent HT Detection from Fill (Pos: equivHT, Neg: Fill)
Model       |    Detection |          FPR |     Accuracy 
---------------------------------------------------------
fold1       |       0.9464 |       0.0000 |       0.9946 
fold2       |       0.9821 |       0.0000 |       0.9982
```

Key insight: The model trained on fold1 exhibits a performance degradation due to the presence of unknown HT pixels during training. However, models trained on uncontaminated folds (e.g., fold2) effectively mitigate this drop, validating the fold-rotation strategy.



## Project Structure

```
├── main.py                    Training entry point
├── model.py                   Siamese network architecture
├── online_data_prepare.py     Online data preparation + equivHT coordinate generation
├── offline_data_generator.py  Offline data generation from raw data
├── evaluate.py                Cross-fold evaluation
├── utils.py                   Utility functions
├── data/
│   └── offline_data/          Pre-generated offline data (included)
│   └── online_data/           Original data
└── weights/
    └── pre/                   Pre-trained models (included)
    └── test_train/            test-trained models on Offline data

```
