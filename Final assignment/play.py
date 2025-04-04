import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import torch
from transformers import pipeline
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize
from imageio import imread, imwrite 
import cv2

# Define the custom transformation class
class CustomTransform:
    def __init__(self):
        self.image_transform = Compose([
            ToImage(),  # Convert PIL image to tensor (CxHxW)
            ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0,1]
            #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize for RGB
        ])

    def add_fog(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", use_fast=True)

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

    def add_night(self,img):
        """
        Apply a night-time effect to an image by reducing brightness, 
        adding a blue tint, and increasing contrast.
        """
        # Reduce brightness (make it darker)
        enhancer = ImageEnhance.Brightness(img)
        img_night = enhancer.enhance(0.5)  # Lower brightness significantly
        
        # Add a blue tint (simulate moonlight)
        img_array = np.array(img_night).astype(np.float32)  # Convert to float for operations
        img_array[:, :, 0] *= 0.7  # Reduce red channel
        img_array[:, :, 1] *= 0.8  # Reduce green channel
        img_array[:, :, 2] *= 1.15  # Boost blue channel
        
        # Clip values to valid range (0-255)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Convert back to an image
        img_night = Image.fromarray(img_array)
        
        # Increase contrast to make artificial lights stand out
        enhancer = ImageEnhance.Contrast(img_night)
        img_night = enhancer.enhance(1.5)  # Adjust contrast

        return img_night

    def __call__(self, img, mode):
        # Apply fog effect on the image
        if mode=="night":
            img = self.add_night(img)
        if mode=="fog":
            img = self.add_fog(img)
        img = self.image_transform(img)  # Apply normalization and resize
        return img

def visualize_transform(original_img, fog_img, night_img):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original_img = np.array(original_img)
    fog_img = fog_img.permute(1, 2, 0).numpy()  #
    night_img = night_img.permute(1, 2, 0).numpy()  #

    # Convert images to numpy for display
    axes[0].imshow(np.array(original_img))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(np.array(fog_img))
    axes[1].set_title("Fog Effect")
    axes[1].axis('off')

    axes[2].imshow(np.array(night_img))
    axes[2].set_title("Night Effect")
    axes[2].axis('off')

    plt.show()

# Load a PNG image (replace 'your_image.png' with the correct path)
img_path = r'C:\Users\20193857\Documents\Visual_Studio_2017\NeuralNetworks\NNCV\Final assignment\test.png'
original_img = Image.open(img_path).convert('RGB')
# Initialize the custom transform
transform = CustomTransform()
# Apply transformation
fog_img = transform(original_img,"fog")
night_img = transform(original_img,"night")
# Visualize the original and transformed images
visualize_transform(original_img, fog_img, night_img)
