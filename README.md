# 🧠 Age Classification via Regularized ResNet-18

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Architecture-ResNet--18-blueviolet?style=for-the-badge"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Phase 1 — Knowledge Distillation](#-phase-1--knowledge-distillation)
- [Phase 2 — Strong Regularization + EMA](#-phase-2--strong-regularization--ema)
- [Model Comparison Results](#-model-comparison-results)
- [Evaluation Framework](#-evaluation-framework)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Assignment Constraints](#-assignment-constraints)
- [References](#-references)

---

## 🔍 Overview

This project builds a binary image classifier that predicts whether a face photograph depicts a **young** (class 0) or **old** (class 1) person. The work is split into two phases:

| Phase | Core Strategy | Key Innovations |
|-------|--------------|-----------------|
| **Phase 1** | Knowledge Distillation from CLIP | Soft-label supervision via OpenAI CLIP, Warm-up Cosine LR, MixUp |
| **Phase 2** | Strong Regularization + EMA | RandAugment, BatchNorm head, EMA weights, OneCycleLR, Test-Time Augmentation |

Both phases use a **ResNet-18 backbone trained entirely from scratch** (`weights=None`) — no pretrained backbone weights are used, fully compliant with assignment constraints.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Total Training Images | 18,332 (after Phase 2 train+val merge: **18,466**) |
| Class Distribution | Balanced — 9,166 Young · 9,166 Old |
| Validation Images | 134 (flat directory) |
| Input Resolution | 256 × 256 pixels (aligned, cropped face images) |
| Label Format | `0` = Young · `1` = Old |

### Directory Structure

```
dataset/
├── train/
│   ├── 0/          # 9,166 images  (class 0 — Young)
│   └── 1/          # 9,166 images  (class 1 — Old)
├── valid/          # 134 images (flat, no sub-folders)
└── valid_labels.csv
```

> **Download**: See the assignment PDF for the dataset link.  
> **Starter Code**: [github.com/Divyaanshmertia/Deep-Learning-2026-Assignment-1](https://github.com/Divyaanshmertia/Deep-Learning-2026-Assignment-1)

---

## 🏗️ Architecture

### Phase 1 Model — `MyAgeClassifier` (Distillation)

```
ResNet-18 Backbone  (weights=None)
        │
        ▼
  [AdaptiveAvgPool]     ← backbone's global pooling
        │
        ▼
  Linear(512 → 256)
  ReLU
  Dropout(0.3)
  Linear(256 → 2)       ← classification logits
```

### Phase 2 Model — `MyAgeClassifier` (Strong Regularization)

```
ResNet-18 Backbone  (weights=None)
        │
        ▼
  [AdaptiveAvgPool]     ← backbone's global pooling
        │
        ▼
  BatchNorm1d(512)      ← stabilizes feature distribution
  Dropout(0.4)
  Linear(512 → 256)
  ReLU
  Dropout(0.2)
  Linear(256 → 2)       ← classification logits
```

**Test-Time Augmentation (Phase 2 only)**: At inference time, predictions from the original and horizontally-flipped image are averaged for more robust outputs.

```python
def forward(self, x):
    if self.training:
        return self.backbone(x)
    out_original = self.backbone(x)
    out_flipped  = self.backbone(torch.flip(x, dims=[3]))
    return (out_original + out_flipped) / 2.0
```

---

## 🔬 Phase 1 — Knowledge Distillation

### Core Idea

A pretrained **CLIP** foundation model (`openai/clip-vit-base-patch32`) is used to generate *soft probability distributions* over each training image using natural language prompts. These soft predictions serve as auxiliary supervision for the ResNet-18 student.

```python
prompts = [
    "A photo of a young person's face",
    "A photo of an old person's face"
]
```

The student is trained with a **mixed loss**:

```
Loss = α · CrossEntropy(logits, hard_labels)
     + (1 − α) · KL_Divergence(student_log_probs ∥ CLIP_soft_probs)

α = 0.85
```

This encourages the ResNet-18 to align its confidence with CLIP's semantic understanding of age, even though CLIP itself is never used for inference.

### Data Augmentation Pipeline

```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
])
```

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 110 |
| LR Schedule | Warm-up Cosine (5 warm-up + 80 cosine epochs) |
| Gradient Clipping | max_norm = 1.0 |
| Label Smoothing | 0.1 |
| MixUp Alpha | 0.2 |
| KD Temperature | α = 0.85 (CE weight) |

### MixUp with Knowledge Distillation

MixUp is applied jointly to both the image pairs **and** their corresponding CLIP soft labels. For a mixed sample with interpolation coefficient λ:

```
mixed_image   = λ · x_a  + (1 − λ) · x_b
teacher_mixed = λ · CLIP(x_a) + (1 − λ) · CLIP(x_b)
Loss          = α · MixupCE(logits, y_a, y_b, λ)
              + (1 − α) · KL(student ∥ teacher_mixed)
```

---

## ⚡ Phase 2 — Strong Regularization + EMA

### Core Improvements over Phase 1

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Head | Dropout → Linear | **BN → Dropout(0.4) → Linear → Dropout(0.2) → Linear** |
| Augmentation | ColorJitter + RandomErasing | **RandAugment + RandomErasing** |
| LR Schedule | Warm-up Cosine | **OneCycleLR** |
| Weight Averaging | None | **Exponential Moving Average (decay=0.9997)** |
| Inference | Single pass | **Test-Time Augmentation (TTA)** |
| CLIP Distillation | ✅ | ❌ (removed for cleaner training) |

### Exponential Moving Average (EMA)

An EMA model (`torch.optim.swa_utils.AveragedModel`) maintains a temporally-smoothed copy of the network weights throughout training. The EMA parameters are saved as the final submission model — they generally exhibit better generalization than the last-epoch weights.

```python
ema_model = swa_utils.AveragedModel(
    model,
    multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.9997)
)
# Updated every batch:
ema_model.update_parameters(model)
```

### OneCycleLR Schedule

The OneCycleLR schedule drives the learning rate through a single cycle — warm-up to `max_lr=3e-3` over 20% of training, then anneal to a flat minimum — encouraging convergence to flatter, more generalizable minima.

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate (initial) | 1e-3 |
| Max LR (OneCycle) | 3e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 80 |
| EMA Decay | 0.9997 |
| LR Schedule | OneCycleLR (pct_start=0.2) |
| Gradient Clipping | max_norm = 1.0 |
| Label Smoothing | 0.1 |
| MixUp Alpha | 0.2 |
| Dataset Size | 18,466 (train + valid combined) |

### Enhanced Augmentation Pipeline

```python
T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=5),   # stochastic geometric + photometric transforms
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2)
])
```

---

## 📊 Model Comparison Results

> Evaluated on a held-out sample from the public **UTKFace** benchmark  
> (1,000 images, `young` if age < 30, else `old`, same indices used for both models)

### Core Metrics

| Metric | Phase 1 · Distillation | Phase 2 · Strong Reg. | Δ |
|--------|:---------------------:|:--------------------:|:--:|
| **Accuracy** | **0.686** | 0.663 | Phase 1 +2.3% |
| **Balanced Accuracy** | **0.687** | 0.664 | Phase 1 +2.3% |
| **Precision** | **0.719** | 0.697 | Phase 1 +2.2% |
| **Recall** | **0.631** | 0.600 | Phase 1 +3.1% |
| **F1 Score** | **0.672** | 0.645 | Phase 1 +2.7% |
| **ROC AUC** | 0.735 | **0.741** | Phase 2 +0.6% |
| **Log Loss** | 0.693 | **0.641** | Phase 2 −0.052 |
| **Inference (ms/img)** | **2.69** | 3.43 | Phase 1 −0.74ms |
| **Confidence Mean** | **82.1%** | 78.3% | Phase 1 +3.8% |
| **Prediction Entropy** | **0.417** | 0.471 | Phase 1 lower |

> **Key Insight**: Phase 1 achieves higher raw accuracy, faster inference, and higher mean confidence. Phase 2 shows better calibration (lower log loss) and higher ROC AUC — meaning it ranks positive/negative samples more reliably even if the accuracy is slightly lower. This is consistent with the stronger regularization producing a more conservative, better-calibrated model.

### Robustness Evaluation

Both models were evaluated against five image perturbations:

| Condition | Description |
|-----------|-------------|
| `clean` | Original images, no perturbation |
| `jpeg_q30` | JPEG compression at quality=30 |
| `gaussian_noise` | Additive Gaussian noise (σ=0.08) |
| `blur_r2` | Gaussian blur (radius=2) |
| `brightness_0.7` | Brightness reduction by 30% |

Run `03_model_comparison.ipynb` to see the full robustness comparison chart.

---

## 🧪 Evaluation Framework

Three dedicated evaluation notebooks are provided:

### `01_phase1_evaluation.ipynb`
Comprehensive evaluation of the Phase 1 distillation model. Exports:
- Core metrics (accuracy, balanced accuracy, precision, recall, F1, ROC AUC, log loss)
- Normalized confusion matrix
- ROC curve and calibration curve
- Confidence and entropy distributions
- Per-age-bucket accuracy (0–9, 10–19, ..., 70+)
- Top-100 error analysis (highest-confidence wrong predictions)
- Robustness under 5 image perturbations
- Inference latency (ms/image)
- Saved outputs → `phase1_outputs/`

### `02_phase2_evaluation.ipynb`
Identical evaluation pipeline for the Phase 2 strong-regularization model.  
Uses the **same 1,000 sample indices** as Phase 1 (loaded from `evaluation_indices.npy`) for a fair apples-to-apples comparison.  
Saved outputs → `phase2_outputs/`

### `03_model_comparison.ipynb`
Head-to-head model comparison notebook. Does **not** reload models — reads exported metrics/predictions CSVs directly:
- Side-by-side bar chart of all core metrics
- Inference latency comparison
- Accuracy-by-age-bucket overlay plot
- Confidence and entropy distribution overlay
- Win/Loss/Agreement breakdown (Both correct · Phase 1 only · Phase 2 only · Both wrong)
- Robustness comparison overlay
- Final saved summary → `model_comparison_table.csv`

---

## 📁 Project Structure

```
age-classification/
│
├── 📓 Training Notebooks
│   ├── part1.ipynb                   # Phase 1: CLIP distillation training
│   └── part2.ipynb                   # Phase 2: Strong regularization + EMA training
│
├── 📓 Evaluation Notebooks
│   ├── 01_phase1_evaluation.ipynb    # Full metric suite for Phase 1 model
│   ├── 02_phase2_evaluation.ipynb    # Full metric suite for Phase 2 model
│   └── 03_model_comparison.ipynb     # Side-by-side Phase 1 vs Phase 2 analysis
│
├── 📊 Results
│   └── model_comparison_table.csv    # Final metrics summary (both phases)
│
├── 📄 Reports
│   ├── part1.pdf                     # Phase 1 one-page submission report
│   └── part2.pdf                     # Phase 2 one-page submission report
│
├── 📄 Assignment
│   └── deep_learning_2026_assigment_1.pdf  # Official assignment specification
│
└── 📦 Submission Files (generated at runtime)
    ├── b23cm1003.py                  # Model class definition
    └── b23cm1003.pth                 # Saved full model (torch.save(model, ...))
```

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install torch torchvision pillow numpy pandas matplotlib seaborn scikit-learn tqdm transformers datasets
```

### Running on Google Colab (Recommended)

Both training notebooks are designed for **Google Colab** with GPU acceleration. Mount your Google Drive for checkpoint persistence:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload or unzip the dataset:

```python
# If dataset.zip is in /content:
!unzip /content/dataset.zip
```

Verify the dataset structure looks like:

```
/content/dataset/
├── train/
│   ├── 0/   # Young images
│   └── 1/   # Old images
├── valid/
└── valid_labels.csv
```

---

## 🚀 Usage

### Phase 1 Training (`part1.ipynb`)

**Step 1**: Generate CLIP soft labels (run once, save to disk):
```python
extract_clip_soft_labels(DATA_DIR="/content/dataset", OUTPUT_PATH="/content/clip_soft_labels.pt")
```

**Step 2**: Set training mode:
```python
FINAL_SUBMISSION_MODE = True   # Merges train + val for final submission
                               # False = train only, monitors validation accuracy
```

**Step 3**: Run the training loop. The best model is saved automatically to `b23cm1003.pth`.

**Step 4**: Download the model:
```python
from google.colab import files
files.download("b23cm1003.pth")
```

---

### Phase 2 Training (`part2.ipynb`)

All training is encapsulated in the `train_model()` function:

```python
train_model(data_dir='/content/dataset')
```

This automatically:
- Builds the combined train+val dataset (18,466 images)
- Initializes the EMA model with decay=0.9997
- Trains for 80 epochs with OneCycleLR
- Saves the **EMA-averaged** model as `b23cm1003.pth`

---

### Evaluating Your Submission

Use the provided evaluation script before submitting:

```bash
python evaluate_submission_student.py \
  --model_path b23cm1003.pth \
  --model_file b23cm1003.py \
  --data_dir dataset/ \
  --valid
```

---

### Running the Evaluation Notebooks

**Phase 1 evaluation**:
1. Upload `part1.pth` to `/content/`
2. Run `01_phase1_evaluation.ipynb` end-to-end
3. Download `phase1_outputs.zip`

**Phase 2 evaluation**:
1. Upload `part2.pth` to `/content/`
2. Upload `evaluation_indices.npy` (generated by notebook 01) to `/content/`
3. Run `02_phase2_evaluation.ipynb` end-to-end
4. Download `phase2_outputs.zip`

**Model comparison**:
1. Upload both `phase1_outputs.zip` and `phase2_outputs.zip` to `/content/`
2. Run `03_model_comparison.ipynb` end-to-end
3. All comparison plots and `model_comparison_table.csv` are generated automatically

---

## 📐 Submission Format

Each phase requires exactly three files named with your roll number:

```
b23cm1003.py    ← Model class definition (importable by torch.load)
b23cm1003.pth   ← Full model saved with torch.save(model, 'b23cm1003.pth')
b23cm1003.pdf   ← One-page approach report
```

> ⚠️ **Critical**: Save the **full model**, not just the state dict:
> ```python
> torch.save(model, 'b23cm1003.pth')     # ✅ Correct
> torch.save(model.state_dict(), ...)    # ❌ Wrong
> ```

---

## 🔒 Assignment Constraints

| Constraint | Phase 1 | Phase 2 |
|-----------|:-------:|:-------:|
| ResNet-18 backbone (unchanged depth/core layers) | ✅ | ✅ |
| Trained from scratch (`weights=None`) | ✅ | ✅ |
| No pretrained backbone used as primary model | ✅ | ✅ |
| Foundation models only as reference (CLIP for soft labels) | ✅ | ✅ |
| Single model only (no ensembles) | ✅ | ✅ |
| Training data: provided dataset only | ✅ | ✅ |

---

## 📚 References

- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
- Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
- Cubuk, E. D. et al. (2020). *RandAugment: Practical Automated Data Augmentation*. CVPR.
- Müller, R. et al. (2019). *When Does Label Smoothing Help?* NeurIPS.
- Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
- Smith, L. N. & Topin, N. (2019). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*.
- [PyTorch Documentation](https://pytorch.org/docs)
- [torchvision Models](https://pytorch.org/vision/stable/models)
- [Starter Repository](https://github.com/Divyaanshmertia/Deep-Learning-2026-Assignment-1)

---

<p align="center">
  Made with ❤️ for Deep Learning Spring 2026 
</p>
