# RTDS: A Robust Two-Stage Tongue Diagnosis System

**Official PyTorch Implementation**

**Authors:** Boyang Zhang (Boston University)  
**Paper:** *RTDS: A Robust Two-Stage Tongue Diagnosis System with U-Net++ Segmentation and Swin-Hybrid Classification* (Submitted to SIVP)

---

## 📖 Introduction

In the field of Traditional Chinese Medicine (TCM), automated tongue diagnosis faces significant challenges due to **uncontrolled imaging environments**. Real-world clinical images are often compromised by:
* **Complex Backgrounds:** Lips, teeth, and facial skin that act as noise.
* **Variable Illumination:** Glare, shadows, and uneven lighting.
* **Label Ambiguity:** Subtle visual differences between diagnostic categories.

To address these issues, we propose **RTDS**, a decoupled deep learning framework that explicitly separates **ROI Extraction** from **Disease Diagnosis**. 

This repository provides the complete source code to reproduce our **Two-Stage Pipeline**:
1.  **Stage 1 (Segmentation):** A pixel-level **U-Net++** segmentation network to automatically remove background noise and generate clean Regions of Interest (ROIs).
2.  **Stage 2 (Classification):** A **Swin-Hybrid** classifier that fuses CNN local texture features with Transformer global context, optimized via **Focal Loss** for robust clinical decision-making.
![Figue1](https://github.com/user-attachments/assets/fbd2e76f-55c4-4961-9fb4-1e2381e63511)

---

## 📂 Project Structure

The project is organized into two independent stages to ensure modularity and data privacy compliance.

```text
.
├── Stage1_segmentation/              # [Stage 1] Background Removal & ROI Extraction
│   ├── train_sota_segmentation.py    # Training script for the U-Net++ segmentation model
│   └── clean_dataset_run.py          # Inference script: Uses trained model to remove backgrounds from the raw dataset
│
│── classification_model.py           # Definition of the Swin-Hybrid backbone
│── train_classification.py           # Main training script for disease diagnosis
├── utils.py                          # Helper functions (Focal Loss, Seed Initialization)
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```
Data Availability & Ethics Statement
Important Note: The dataset used in this study comprises 2,100 clinical tongue images collected from hospital outpatient departments. Due to strict patient privacy regulations and ethical guidelines regarding biometric data, the raw image dataset is NOT publicly available.
However, to guarantee the reproducibility of our proposed method, we have open-sourced the complete code ecosystem. This includes the network architectures, loss functions, and the critical data cleaning pipeline (clean_dataset_run.py), allowing researchers to verify our methodology or apply it to their own private cohorts.


## 📊 Performance Highlights

Our **Two-Stage** strategy significantly outperforms standard end-to-end models on clinical data.

| Method | Backbone | Input Data | Top-1 Accuracy |
| :--- | :--- | :--- | :---: |
| Standard End-to-End | ResNet-34 | Raw Images | 55.48% |
| Two-Stage Baseline | ResNet-34 | **Clean ROIs (Stage 1)** | 66.03% |
| **RTDS (Ours)** | **Swin-Hybrid** | **Clean ROIs (Stage 1)** | **75.76%** |

> **Key Insight:**

> 1. **Stage 1 Matters:** Removing background noise via our U-Net++ module improves accuracy by over **10%** (55.48% → 66.03%).


> 2. **Swin-Hybrid Matters:** Our hybrid architecture further boosts performance by nearly **10%** over the CNN baseline (66.03% → 75.76%).

## Figure
![Fig5](https://github.com/user-attachments/assets/49fa6def-4f79-4ad4-8c65-e0c2830a56ad)

