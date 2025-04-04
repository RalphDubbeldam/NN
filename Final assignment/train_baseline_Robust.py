"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
test
"""
import os
from argparse import ArgumentParser
from PIL import Image, ImageEnhance
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
)
from torchvision.transforms.v2.functional import hflip
from unet_baseline_augm import Model
import torch
import torchvision.transforms.functional as Fv
from torchvision.transforms import (
    Compose, Resize, Normalize, ToTensor
)
import torchvision.transforms.v2.functional as F2
import random
import numpy as np
from transformers import pipeline

class CustomTransform:
    def __init__(self):
        self.image_transform = Compose([
            ToTensor(),
            Resize((256, 256)),  # Resize image
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize for RGB
        ])
        self.label_transform = Resize((256, 256), interpolation=Fv.InterpolationMode.NEAREST)
        
    def add_fog(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device, use_fast=True)

        depth = pipe(img)["depth"]
        # reduce saturation
        enhancer = ImageEnhance.Color(img)
        img_2 = enhancer.enhance(0.6)
        # reduce brightness
        enhancer2 = ImageEnhance.Brightness(img_2)
        img_2 = enhancer2.enhance(0.75)
        # increase contrast
        enhancer3 = ImageEnhance.Contrast(img_2)
        img_2 = enhancer3.enhance(2)
        # Create a white layer with the same size as the input images
        white_layer = Image.new('RGBA', img_2.size, (216,216,216,0))
        # Convert images to numpy arrays for easier manipulation
        grayscale_array = np.array(depth)
        white_array = np.array(white_layer)
        # Set the alpha channel of the white layer
        white_array[:, :, 3] = 255 - grayscale_array
        # Convert back to PIL Image
        white_layer_transparent = Image.fromarray(white_array, 'RGBA')
        # Composite the images
        result = Image.alpha_composite(img_2.convert('RGBA'), white_layer_transparent)
        return result.convert('RGB')


    def __call__(self, img, target):
        # Apply the same random rotation
        angle = random.uniform(-10, 10)  # Generate random angle
        img = Fv.rotate(img, angle)  
        target = Fv.rotate(target, angle, interpolation=Fv.InterpolationMode.NEAREST)  
        # Apply horizontal flip for 50 percent
        if torch.rand(1) < 0.5:
            img = F2.hflip(img)
            target = F2.hflip(target)
        # Apply fog to 10 percent of images
        if torch.rand(1) < 0.1:
            img = self.add_fog(img)        
        img = self.image_transform(img)
        target = self.label_transform(target)
        target = target.to(torch.long)  # Ensure labels are integers
        return img, target

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser

#https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
def multiclass_dice_coefficient(pred, target, smooth=1):
    pred = F.softmax(pred.clone(), dim=1)  # Clone to avoid modifying original tensor

    num_classes = pred.shape[1]
    target = torch.clamp(target, min=0, max=num_classes - 1)  # Avoid invalid indices
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dice = 0
    for c in range(num_classes):
        pred_c = pred[:, c].clone()  # Clone to prevent in-place modification
        target_c = target_one_hot[:, c]

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice += (2. * intersection + smooth) / (union + smooth)

    return dice.mean() / num_classes


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    transform = CustomTransform()

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            Dices = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                Dice = multiclass_dice_coefficient(outputs, labels)  # Compute Dice Loss
                losses.append(loss.item())
                Dices.append(Dice)
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            valid_Dice = sum(Dices) / len(Dices)
            wandb.log({
                "valid_loss": valid_loss,
                "valid_DiceCoefficient": valid_Dice
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
