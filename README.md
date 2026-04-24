# FANR

This repository contains the implementation of **FANR**, a frequency decomposition INR framework for scientific volume data representation.

## Repository Structure

```text
FANR/
├── INR.py
├── OURS.py
├── utils.py
├── INR_base/
│   ├── load_data.py
│   ├── model.py
│   └── train.py
├── OURS_base/
│   ├── load_data.py
│   ├── model.py
│   └── train.py
└── dataset/
```

## Code Description

### `utils.py`

`utils.py` contains basic utility functions shared by different methods, such as dataset selection, learning rate scheduling, and other common training utilities.

### `INR_base/`

`INR_base/` contains the implementation of the baseline INR methods used for comparison.

- `load_data.py`: data loading and preprocessing for baseline methods.
- `model.py`: model definitions of the compared INR methods, including **CoordNet**, **KD-INR**, **FINER**, and **NGP**.
- `train.py`: data loading, training, inference, and output generation for the baseline methods.

### `OURS_base/`

`OURS_base/` follows a similar structure to `INR_base/`, but implements our proposed method.

- `load_data.py`: data loading and preprocessing for our method.
- `model.py`: model definition of our frequency decomposition INR framework.
- `train.py`: data loading, training, inference, and output generation for our method.

### `INR.py`

`INR.py` is the main script for training and inference of the baseline methods.

Example usage:

```bash
python INR.py --dataset vortex --model KDINR --mode train
```

For evaluation or inference only:

```bash
python INR.py --dataset vortex --model KDINR --mode eval
```

### `OURS.py`

`OURS.py` is the main script for training and inference of our proposed method.

Example usage:

```bash
python OURS.py --dataset vortex --mode train
```

For evaluation or inference only:

```bash
python OURS.py --dataset vortex --mode eval
```

## Supported Baselines

The following baseline methods are implemented in `INR_base/model.py`:

- CoordNet
- KD-INR
- FINER
- NGP

## Dataset

The `dataset/` folder contains example volume data used by the scripts. The raw volume files are read during training and inference according to the dataset name specified by the command line arguments.

## Outputs

The training and inference outputs are saved under the result directory specified in the scripts. Please check the corresponding output folders after running training or inference.

## Environment

The experiments were conducted on a Linux workstation with an NVIDIA GeForce RTX 4090 GPU.

Main environment:

- Python 3.8.19
- PyTorch 2.1.2 + CUDA 12.1
- NVIDIA Driver 550.100
- GPU: NVIDIA GeForce RTX 4090, 24 GB memory
- NumPy 1.24.4
- SciPy 1.10.1
- scikit-image 0.21.0
- Matplotlib 3.7.5
- tqdm 4.66.4
- tiny-cuda-nn

The code may also run under other compatible CUDA/PyTorch environments.
