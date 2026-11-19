---
layout: default
title: Installation
---

# Installation Guide

## Prerequisites
- Linux / macOS (System calls rely on standard Unix tools like `sed`, `awk`, `grep`)
- Python 3.9
- Conda / Mamba
- ViennaRNA Package (`RNAduplex`)

## Quick Setup

```bash
git clone [https://github.com/your-username/plant-trf-target-finder.git](https://github.com/your-username/plant-trf-target-finder.git)
cd plant-trf-target-finder
conda env create -f environment.yml
conda activate trf_target_finder