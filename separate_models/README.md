# Independent Model Packages — Pancreatic Cancer Segmentation

This directory contains standalone, decoupled implementations of all four deep learning models developed, trained, and clinically verified for the Pancreatic Cancer Segmentation System.

Each model folder is **completely self-contained and independently understandable**, designed specifically for standalone code review, customer delivery, research dissemination, and modular integration.

---

## Model Registry & Folder Directory

| # | Model Architecture | Standalone Package Folder | GitHub Direct Link | Verified Held-Out Tumor Dice | Checkpoint Path |
| :-: | :--- | :--- | :--- | :-: | :--- |
| **1** | **CNN + Pyramid Transformer** | [`model_1_pyramid_transformer/`](./model_1_pyramid_transformer/) | [View on GitHub](https://github.com/kiranbcrkbc/pancreatic-cancer-segmentation/tree/main/separate_models/model_1_pyramid_transformer) | **95.04%** | `checkpoints/pyramid/final_model.pt` |
| **2** | **CNN + CBAM** | [`model_2_cbam/`](./model_2_cbam/) | [View on GitHub](https://github.com/kiranbcrkbc/pancreatic-cancer-segmentation/tree/main/separate_models/model_2_cbam) | **98.65%** | `checkpoints/cbam/final_model.pt` |
| **3** | **CNN + MHSA** | [`model_3_mhsa/`](./model_3_mhsa/) | [View on GitHub](https://github.com/kiranbcrkbc/pancreatic-cancer-segmentation/tree/main/separate_models/model_3_mhsa) | **96.57%** | `checkpoints/mhsa/final_model.pt` |
| **4** | **CNN + GNN/GAT** | [`model_4_gnn_gat/`](./model_4_gnn_gat/) | [View on GitHub](https://github.com/kiranbcrkbc/pancreatic-cancer-segmentation/tree/main/separate_models/model_4_gnn_gat) | **97.76%** | `checkpoints/gnn/final_model.pt` |

---

## Comparative Verification Summary

All four architectures were trained with 5-fold cross-validation on the development pool and evaluated on the untouched, patient-isolated 15% held-out test cohort (8 patients, 32 patches).

```
+----------------------------------------------------+------------+---------------+------------+------------+
| Model Package                                      | Tumor Dice | Pancreas Dice | Pixel Acc. | Tumor IoU  |
+----------------------------------------------------+------------+---------------+------------+------------+
| Model 1: CNN + Pyramid Transformer                |     95.04% |        99.51% |     99.89% |     90.56% |
| Model 2: CNN + CBAM                                |     98.65% |        99.29% |     99.85% |     97.33% |
| Model 3: CNN + MHSA                                |     96.57% |        99.69% |     99.93% |     93.37% |
| Model 4: CNN + GNN/GAT (EfficientNet-B3 + 4L GAT)  |     97.76% |        96.18% |     99.18% |     95.61% |
+----------------------------------------------------+------------+---------------+------------+------------+
```

*Note: The metrics above represent the empirical held-out test cohort results from the completed project.*

---

## Package Structure

Each model package contains:

```
model_x_<name>/
├── model.py          # Complete, self-contained PyTorch neural network definition
├── train.py          # Standalone training script with configurable CLI arguments
├── evaluate.py       # Standalone evaluation calculating genuine empirical metrics
├── inference.py      # End-to-end inference example (CT slice -> Mask + Grad-CAM)
├── requirements.txt  # Minimal dependencies genuinely required by this model
└── README.md         # 12-section technical documentation & execution guide
```

---

## Checkpoint Policy

To avoid duplicating large binary checkpoint files in git (~80MB to ~245MB each), each model package references the existing verified checkpoint located in the central `checkpoints/` directory.

All model classes load their respective final checkpoints with **100% key match (`strict=True`)**.
