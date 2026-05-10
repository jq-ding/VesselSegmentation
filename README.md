# Multichannel Deep Neural Network for Retinal Vessel Segmentation

> **A Multichannel Deep Neural Network for Retina Vessel Segmentation via a Fusion Mechanism**
> 
> Jiaqi Ding, Zehua Zhang, Jijun Tang, Fei Guo
> 
> *Frontiers in Bioengineering and Biotechnology*, 2021
> [Paper](https://www.frontiersin.org/articles/10.3389/fbioe.2021.697915/full)

## Overview

Retinal vessel segmentation is a fundamental task in computer-aided diagnosis of vascular diseases, but thin and peripheral vessels are notoriously difficult to segment accurately. This work proposes a **multi-channel deep neural network with a fusion mechanism** that explicitly handles thin vessels and thick vessels as separate sub-tasks before combining them.

**Key ideas**:
- Apply U-Net to original images, thin-vessel labels, and thick-vessel labels in parallel as a multi-objective optimization
- Use a dedicated fusion mechanism to combine the three predicted probability maps into a final binary segmentation
- Apply focal loss to handle the strong class imbalance between vessel and background pixels

The method achieves state-of-the-art F1-scores on multiple public benchmarks: **0.8201 on DRIVE** and **0.8239 on STARE**, with particularly strong performance on thin and peripheral vessels.

## Repository Structure

```
VesselSegmentation/
├── lib/                    # Network and loss function definitions
├── prepare_datasets.py     # Convert images to .hdf5 format
├── Thin_ThickLabel.py      # Generate thin/thick vessel labels via skeletonization
├── training.py             # Training script
├── predict.py              # Prediction with AUC computation
├── merge.py                # Fusion of three probability maps
├── pixel.py                # Compute Acc, Sp, Se, F1 metrics
└── configuration.txt       # Training configuration
```

## Datasets

- **DRIVE**: https://drive.grand-challenge.org/
- **STARE**: https://cecas.clemson.edu/~ahoover/stare/
- **IOSTAR**: http://www.retinacheck.org/

## Setup

The codebase uses Keras/TensorFlow. Install dependencies:

```bash
conda create -n vessel python=3.8
conda activate vessel
pip install tensorflow keras numpy scikit-image opencv-python h5py
```

## Pipeline

### Step 1: Prepare data

```bash
python prepare_datasets.py
```

Converts source images and labels into `.hdf5` files for efficient loading.

### Step 2: Generate thin and thick vessel labels

```bash
python Thin_ThickLabel.py
```

Extracts the vessel skeleton and separates each ground-truth segmentation into thin- and thick-vessel sub-labels.

### Step 3: Train

Train three U-Net branches separately (original, thin, thick):

```bash
python training.py
```

The training uses a focal loss formulation to address class imbalance:

`loss = -alpha * (1 - p_t)^gamma * log(p_t)`

with default `gamma=2.0` and `alpha=0.25`.

### Step 4: Predict

```bash
python predict.py
```

Outputs per-branch predictions and computes AUC within the field-of-view mask.

### Step 5: Fuse the three predictions

```bash
python merge.py
```

The fusion rule combines original, thick, and thin probability maps into a final binary segmentation.

### Step 6: Compute metrics

```bash
python pixel.py
```

Reports accuracy, specificity, sensitivity, and F1-score.

## Results

The method outperforms many strong baselines on DRIVE, STARE, and IOSTAR, with especially large gains on thin vessels and vascular endpoints — see the paper for full quantitative results and qualitative comparisons.

## Citation

```bibtex
@article{ding2021multichannel,
  title     = {A multichannel deep neural network for retina vessel segmentation via a fusion mechanism},
  author    = {Ding, Jiaqi and Zhang, Zehua and Tang, Jijun and Guo, Fei},
  journal   = {Frontiers in Bioengineering and Biotechnology},
  volume    = {9},
  pages     = {697915},
  year      = {2026},
  publisher = {Frontiers Media SA}
}
```

## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
