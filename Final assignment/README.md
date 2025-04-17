
# Semantic Segmentation on Cityscapes Dataset

This project focuses on training various semantic segmentation models using the Cityscapes dataset. The models incorporate 3D input, data augmentations, robust data conditions (night/fog), and custom loss functions to enhance performance and robustness.

---

## Required Libraries

### Standard Library
- os
- random
- argparse
- multiprocessing

### Numeric and Image Processing
- numpy
- PIL (Pillow)

### PyTorch Modules
- torch
- torchvision

### Hugging Face Transformers
- transformers

### Weights and Biases
- wandb for experiment tracking

---

## Cityscapes Dataset Installation

1. Register and Download
   - Visit: https://www.cityscapes-dataset.com/downloads/
   - Register or log in.
   - Download the following files:
     - leftImg8bit_trainvaltest.zip (contains images)
     - gtFine_trainvaltest.zip (contains semantic labels)
   - Extract the files into a directory of your choice.

---

## Running the Project on SLURM

The `jobscript_slurm.sh` file includes SLURM directives for job submission:

```
#SBATCH --nodes=1             # Number of compute nodes
#SBATCH --ntasks=1            # Number of tasks
#SBATCH --cpus-per-task=18    # CPU cores per task
#SBATCH --gpus=1              # Request one GPU
#SBATCH --partition=gpu_a100  # Partition with A100 GPUs
#SBATCH --time=00:30:00       # Job time limit (HH:MM:SS)
```

---

## How to Run a Model

In `main.sh`, specify your model configuration:

```
python3 train_combined_Robust.py \
    --data-dir ./data/cityscapes \      # Path to Cityscapes data
    --batch-size 64 \                   # Batch size
    --epochs 100 \                      # Number of training epochs
    --lr 0.001 \                        # Learning rate
    --num-workers 10 \                  # Number of data loader workers
    --seed 42 \                         # Random seed
    --experiment-id "Robust"            # Experiment name (for wandb)
```

Change `train_combined_Robust.py` to the desired model script (see below).

---

## Model Variants

| Model Name         | Description |
|--------------------|-------------|
| baseline           | Basic model using RGB input and cross-entropy loss. |
| baseline_3D        | Adds depth input to baseline. |
| baseline_augm      | Applies data augmentation (rotation, horizontal flip, normalization). |
| baseline_loss      | Uses a combined Dice and Cross-Entropy loss. |
| baseline_Robust    | Adds augmentations + robust conditions (night/fog) to baseline. |
| combined           | Combines 3D input with Dice-CE loss for better performance. |
| combined_Robust    | Fully enhanced model: 3D input, Dice-CE loss, augmentations, and night/fog. |

---

## User Info

- Codalab Username: RalphDubbeldam
- TU/e Email: r.a.dubbeldam@student.tue.nl

---

Feel free to reach out if you encounter issues or have suggestions.
