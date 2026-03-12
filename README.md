# GDFN & HFFN — Dual-Branch Networks for Autism Spectrum Disorder Screenign

Two dual-branch deep learning architectures are proposed:
- **GDFN** — fuses CNN/Transformer image features with facial *landmark geometric distances* (31 pairs, dlib 68-point model)
- **HFFN** — fuses CNN/Transformer image features with *SIFT descriptor* (25 600-d feature vector)

Both networks are evaluated on two datasets and with seven pretrained backbones.

---

## Repository Structure

```
├── GDFN/                              # Geometric Distance Feature Network
│   ├── GDFN_AID.py                   # TF  | AID dataset  | train/valid/test
│   ├── GDFN_AID_KFold.py             # TF  | AID dataset  | 5-Fold CV
│   ├── GDFN_Attention.py             # TF  | Attention dataset | 5-Fold CV
│   ├── GDFN_ViT_Swin_AID.py          # PyTorch | AID | ViT-B/16, Swin-B
│   ├── GDFN_ViT_Swin_AID_KFold.py   # PyTorch | AID | ViT-B/16, Swin-B | 5-Fold CV
│   └── GDFN_ViT_Swin_Attention.py   # PyTorch | Attention | ViT-B/16, Swin-B | 5-Fold CV
│
├── HFFN/                              # Hybrid Feature Fusion Network
│   ├── HFFN_AID.py                   # TF  | AID dataset  | train/valid/test
│   ├── HFFN_AID_KFold.py             # TF  | AID dataset  | 5-Fold CV
│   ├── HFFN_Attention.py             # TF  | Attention dataset | 5-Fold CV
│   ├── HFFN_ViT_Swin_AID.py          # PyTorch | AID | ViT-B/16, Swin-B
│   ├── HFFN_ViT_Swin_AID_KFold.py   # PyTorch | AID | ViT-B/16, Swin-B | 5-Fold CV
│   └── HFFN_ViT_Swin_Attention.py   # PyTorch | Attention | ViT-B/16, Swin-B | 5-Fold CV
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Architecture Overview

### GDFN — Geometric Distance Feature Network

```
Input Image ──► Pretrained Backbone ──► Flatten ──► Dense(512) ──► Dense(256) ──┐
                                                                                  ├──► Concat ──► Dense(256) ──► Dense(2)
Input Image ──► 31 Distances ─────────────────────► Dense(256) ──► Dense(128) ──┘
```

The 31 distance pairs are computed from 18 selected dlib facial landmarks covering eyes, nose, and mouth regions.

### HFFN — Histogram Feature Fusion Network

```
Input Image ──► Pretrained Backbone ──► Flatten ──► Dense(512) ──► Dense(256) ──┐
                                                                                  ├──► Concat ──► Dense(256) ──► Dense(2)
Input Image ──► SIFT (25600) ─────────────────────► Dense(256) ──► Dense(128) ──┘
```

SIFT keypoint descriptors are extracted and fixed to a 25 600-dimensional vector.

### Backbones Evaluated

| Framework   | Backbones                                                    |
|-------------|--------------------------------------------------------------|
| TensorFlow  | MobileNetV1, MobileNetV2, InceptionV3, DenseNet121, Xception |
| PyTorch     | ViT-B/16, Swin-B                                             |

**Training:** Adam (lr = 1e-4), batch = 32, **epochs = 70**
**Augmentation (KFold variants):** Random rotation ±90°, zoom 20%, horizontal flip

---
## Datasets
### Attention Dataset
- Source: Our VR-CPT Task
- Classes: `ASD` | `TD` (Typically Developing)
- Directory: `test-face-vs1/`
- Split: 80% train (5-Fold CV) / 20% test


### AID — Autism Image Dataset
- Source: [Autism Image Dataset](https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set)
- Classes: `autistic` | `non_autistic`
- Split: `AID/train/` | `AID/valid/` | `AID/test/`



> **Note:** Datasets are **not included** in this repository. Download and place them in the folder structure shown above before running.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `opencv-contrib-python` is required for SIFT features (HFFN scripts).
> For GDFN-only use, `opencv-python` is sufficient.

### 4. Download dlib shape predictor
Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the **root** of the repository (same level as `GDFN/` and `HFFN/`).

```
<repo-root>/
├── shape_predictor_68_face_landmarks.dat   ← place here
├── GDFN/
└── HFFN/
```

### 5. Prepare datasets
```
<repo-root>/
├── AID/
│   ├── train/
│   │   ├── autistic/
│   │   └── non_autistic/
│   ├── valid/
│   │   ├── autistic/
│   │   └── non_autistic/
│   └── test/
│       ├── autistic/
│       └── non_autistic/
└── test-face-vs1/
    ├── ASD/
    └── TD/
```

---

## Usage

### TensorFlow variants (CNN backbones)

```bash
# GDFN on AID dataset (train/valid/test split)
python GDFN/GDFN_AID.py

# GDFN on AID with 5-Fold Cross-Validation
python GDFN/GDFN_AID_KFold.py

# GDFN on Attention dataset with 5-Fold CV
python GDFN/GDFN_Attention.py

# HFFN variants (same pattern)
python HFFN/HFFN_AID.py
python HFFN/HFFN_AID_KFold.py
python HFFN/HFFN_Attention.py
```

### PyTorch variants (ViT-B/16 & Swin-B backbones)

```bash
# GDFN with ViT-B/16 and Swin-B on AID dataset
python GDFN/GDFN_ViT_Swin_AID.py

# GDFN with ViT-B/16 and Swin-B — 5-Fold CV
python GDFN/GDFN_ViT_Swin_AID_KFold.py

# GDFN on Attention dataset — 5-Fold CV
python GDFN/GDFN_ViT_Swin_Attention.py

# HFFN variants (same pattern)
python HFFN/HFFN_ViT_Swin_AID.py
python HFFN/HFFN_ViT_Swin_AID_KFold.py
python HFFN/HFFN_ViT_Swin_Attention.py
```

All scripts print per-epoch training/validation accuracy, and at the end output:
- Confusion matrix
- Accuracy, Sensitivity (Recall), Specificity
- Full classification report (Precision, Recall, F1)

---

## Evaluation Metrics

| Metric      | Formula                          |
|-------------|----------------------------------|
| Accuracy    | (TP + TN) / (TP + TN + FP + FN) |
| Sensitivity | TP / (TP + FN)                   |
| Specificity | TN / (TN + FP)                   |
| F1-Score    | 2 × Precision × Recall / (P + R) |




---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
