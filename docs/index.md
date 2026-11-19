---
layout: default
title: Home
---

# Plant tRF Target Finder

Welcome to the documentation for the **Plant tRF Target Finder** tool. This framework provides an end-to-end machine learning pipeline to identify target mRNAs sliced or regulated by plant tRFs.

---

## 🚀 Key Steps in the Pipeline

1. **Dataset Construction:** Merges degradome-supported positive interactions with dinucleotide-aware negative sequence pairs.
2. **Feature Extraction (`Rnaduplex_server.py`):** Calculates structural, sequence, and energy dynamics via ViennaRNA (`RNAduplex`).
3. **Model Training (`model.py`):** Evaluates models across 10-fold cross-validation metrics (MCC, PR-AUC, ROC-AUC, F1).
4. **Hyperparameter Tuning (`tuning.py`):** Optimizes decision boundaries using Bayesian optimization.

---

## 📄 Documentation Links

- [Installation & Setup](installation.html)
- [Input/Output Specifications](usage.html)
- [View Code on GitHub](https://github.com/your-username/plant-trf-target-finder)