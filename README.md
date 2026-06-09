# Compressed-Video-Quality-Enhancement-Based-on-Optimal-Reference-Selection
Compressed Video Quality Enhancement Based on Optimal Reference Selection
## Model Weights

The pretrained model weights are publicly available at [Baidu Netdisk](https://pan.baidu.com/s/1szOT3z3WnCxW2BcG7bON8Q?pwd=5j4y) with the extraction code `5j4y`.

## Dataset

The VCP dataset can be obtained from the [CPGA repository](https://github.com/VQE-CPGA/CPGA).
Due to the large scale of the extended MFQE 2.0 datasets generated with different coding configurations, GOP structures, and codecs, we plan to make them publicly available in the future.
# ORS-CVE

Official implementation of **Compressed Video Quality Enhancement Based on Optimal Reference Selection**.

ORS-CVE is a compressed video enhancement framework that introduces an optimal reference selection strategy and bitstream-prior-guided restoration for improving compressed video quality.

## Overview

Compressed video enhancement aims to reduce compression artifacts and recover fine details from decoded low-quality videos. Existing multi-frame enhancement methods usually use neighboring frames as references, which may introduce redundant information and propagate coding errors along the prediction chain.

ORS-CVE addresses this issue by selecting a more suitable reference frame according to the coding prediction structure. The selected reference frame, motion-vector prior, and residual prior are jointly used to guide video quality enhancement.

## Repository Structure

```text
ORSCVE/
├── README.md
└── ORS_CVE_main/
    ├── config.yml
    ├── train_ors.py
    ├── test_ors.py
    ├── dataset/
    ├── models/
    ├── ops/
    ├── utils/
    └── png/
```

## Requirements

The code is implemented with Python and PyTorch.

We recommend creating a new conda environment:

```bash
conda create -n orscve python=3.8
conda activate orscve
```

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pillow tqdm pyyaml flow-vis
```

If your CUDA version is different, please install the corresponding PyTorch version from the official PyTorch website.

## Dataset

We use the VCP dataset for training and evaluation. The VCP dataset can be obtained from the [CPGA repository](https://github.com/VQE-CPGA/CPGA).

Due to the large scale of the extended MFQE 2.0 datasets generated with different coding configurations, GOP structures, and codecs, we plan to make them publicly available in the future.

Before training or testing, please organize the dataset and bitstream priors according to the dataset loading scripts in `ORS_CVE_main/dataset/`.

A typical dataset structure is as follows:

```text
VCP_dataset/
├── GT/
├── LQ_Priors/
│   └── LD/
│       └── qp37/
└── test_18_data/
    ├── GT/
    └── LD/
        └── qp37/
```

The low-quality folder should contain the decoded compressed frames and the corresponding coding priors required by the dataset loader, including the selected ORS reference, motion-vector prior, and residual prior.

## Model Weights

The pretrained model weights are publicly available at [Baidu Netdisk](https://pan.baidu.com/s/1szOT3z3WnCxW2BcG7bON8Q?pwd=5j4y) with the extraction code `5j4y`.

After downloading the weights, please place them in a suitable folder, for example:

```text
ORS_CVE_main/exp/vcp_ldb_qp37/
```

## Training

Enter the code directory:

```bash
cd ORS_CVE_main
```

Before training, modify the training and validation dataset paths in `train_ors.py` according to your local environment.

For example, modify the following paths:

```python
train_ds = TrainSet(
    lq_root="/path/to/VCP_dataset/LQ_Priors/LD/qp37",
    gt_root="/path/to/VCP_dataset/GT",
    gop=8,
)

val_ds = ValSet(
    lq_root="/path/to/VCP_dataset/test_18_data/LD/qp37",
    gt_root="/path/to/VCP_dataset/test_18_data/GT",
    gop=8,
)
```

You can also modify the training settings in `config.yml`, such as experiment name, batch size, number of iterations, validation interval, learning rate, and GOP size.

Run training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ors.py --opt_path config.yml
```

The checkpoints and logs will be saved under:

```text
ORS_CVE_main/exp/
```

For example:

```text
ORS_CVE_main/exp/vcp_ldb_qp37/
├── log.log
├── ckp_ldb_qp37.pt
└── val_metric_log.txt
```
## Testing

Enter the code directory:

```bash
cd ORS_CVE_main
```

Run testing with a pretrained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ors.py \
  --ckpt exp/vcp_ldb_qp37/ckp_ldb_qp37.pt \
  --lq_root /path/to/VCP_dataset/test_18_data/LD/qp37 \
  --gt_root /path/to/VCP_dataset/test_18_data/GT \
  --gop 8 \
  --batch_size 1 \
  --num_workers 8
```

The script reports the PSNR and SSIM results of the compressed input and the enhanced output.


## Evaluation Output

During testing, the script prints per-video and average results, including:

```text
PSNR of compressed input
PSNR of enhanced output
Delta PSNR
SSIM of compressed input
SSIM of enhanced output
Delta SSIM
```

## Notes

1. The current training script uses local dataset paths inside `train_ors.py`. Please update these paths before training.

## Acknowledgement

This repository is built upon prior works on compressed video quality enhancement. We thank the authors of the VCP and CPGA projects for providing useful datasets and code resources.
