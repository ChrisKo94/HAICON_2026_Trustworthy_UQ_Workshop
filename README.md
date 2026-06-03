# 🔬 Uncertainty Quantification for Deep Learning — Practical Workshop

A 1-hour hands-on workshop covering the theory and practice of uncertainty quantification (UQ) in deep learning, from toy regression examples to real-world Earth Observation applications.

---

## 📚 Notebook Overview

| # | Notebook | Topics | Runtime | Link to Colab |
|---|----------|--------|---------| -------------|
| 01 | **Regression UQ** | Gaussian NLL, Calibration, MC Dropout, Quantile Regression, Ensembles, SNGP | ~20 min | https://colab.research.google.com/github/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/01_regression_uq.ipynb |
| 02 | **Classification UQ** | Softmax overconfidence, Temperature Scaling, MC Dropout, SNGP, OOD Detection | ~20 min | https://colab.research.google.com/github/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/02_mnist_classification_uq.ipynb |
| 03 | **EO / EuroSAT UQ** | Real CNN, Synthetic clouds, Aleatoric uncertainty from distribution shift | ~25 min | https://colab.research.google.com/github/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/03_eo_eurosat_uq.ipynb |
| 04 | **Lightning-UQ-Box** | Automatic pipelines, copy-paste ready implementations | ~15 min | https://colab.research.google.com/github/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/04_lightning_uq_box.ipynb |

---

## 🧠 Key Concepts

### Types of Uncertainty
```
              ┌─────────────────────────────────────────┐
              │           Total Uncertainty              │
              └───────────────┬─────────────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
   ┌────────┴─────────┐              ┌───────────┴────────┐
   │    Aleatoric     │              │     Epistemic       │
   │  (data noise)    │              │  (model ignorance)  │
   │                  │              │                     │
   │  Irreducible     │              │  Reducible with     │
   │  Can model it    │              │  more data          │
   └──────────────────┘              └─────────────────────┘
```

### Methods Covered

| Method | Aleatoric | Epistemic | Cost | Key Idea |
|--------|:---------:|:---------:|:----:|----------|
| Plain MSE | ❌ | ❌ | Low | Baseline, overconfident |
| Gaussian NLL | ✅ | ❌ | Low | Predict mean + variance |
| Laplace/Student-t NLL | ✅ | ❌ | Low | Heavy-tailed noise |
| Quantile Regression | ✅ | ❌ | Low | Pinball loss, no assumptions |
| MC Dropout | ✅ | ✅ | Low | Dropout at test time |
| Deep Ensembles | ✅ | ✅ | High | M independent models |
| SNGP | ✅ | ✅ | Med | Spectral norm + GP head |
| Lightning-UQ-Box | ✅ | ✅ | Auto | All-in-one framework |

### Check out: Lightning-UQ-Box

A new open-source library for easy UQ in PyTorch Lightning, with copy-paste ready implementations of all methods covered in this workshop and more!

- Easy-to-use API for training and evaluating UQ models
- Supports regression, classification, and segmentation tasks
- Built on top of PyTorch Lightning for scalability and flexibility

https://github.com/lightning-uq-box/lightning-uq-box

## ⚙️ Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate uq-workshop
python -m ipykernel install --user --name uq-workshop --display-name "Python 3 (uq-workshop)"
jupyter lab
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name uq-workshop
jupyter lab
```

### Option C: Google Colab
Each notebook starts with a `!pip install` cell — just run it and go.

---

## 🗂️ Repository Structure

```
uq-workshop/
├── notebooks/
│   ├── 01_regression_uq.ipynb        # Regression UQ from scratch
│   ├── 02_classification_uq.ipynb    # Classification UQ + OOD
│   ├── 03_eo_eurosat_uq.ipynb        # EuroSAT + synthetic clouds
│   └── 04_lightning_uq_box.ipynb     # lightning-uq-box framework
├── utils/
│   ├── plotting.py                   # Shared plotting utilities
│   └── calibration.py                # Calibration metrics
├── solutions/                        # Completed notebooks (instructors)
├── figures/                          # Generated figures
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 📖 Further Reading

- [Deep Ensembles (Lakshminarayanan et al., 2017)](https://arxiv.org/abs/1612.01474)
- [SNGP (Liu et al., 2020)](https://arxiv.org/abs/2006.10108)
- [A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/abs/2107.03342)
- [Calibration of Modern Neural Networks (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)
- [lightning-uq-box documentation](https://lightning-uq-box.readthedocs.io/)
- [TorchUncertainty](https://torch-uncertainty.github.io/)
