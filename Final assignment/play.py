import torch
import random
import matplotlib.pyplot as plt
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torchvision.transforms.v2 import Compose, ToTensor, Resize, Normalize
import torchvision.transforms.functional as Fv
from PIL import Image
from PIL import Image, ImageEnhance

# Define the custom transformation class
class CustomTransform:
    def __init__(self):
        self.image_transform = Compose([
            ToTensor(),
            Resize((256, 256)),  # Resize image
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize for RGB
        ])
        self.label_transform = Resize((256, 256), interpolation=Fv.InterpolationMode.NEAREST)
        # Fog effect parameters
        self.fog_intensity = 0.3
        self.fog_color = (200, 200, 200)
        
    def add_fog(self, img):
        """
        Adds a fog effect to the image by blending with a semi-transparent fog layer.
        :param img: PIL Image to which fog will be applied.
        :return: Fogged image (PIL Image).
        """
        # Generate a fog image (a solid color image)
        fog = Image.new('RGB', img.size, self.fog_color)
        
        # Adjust fog intensity
        fog = ImageEnhance.Brightness(fog).enhance(self.fog_intensity)

        # Randomly blend fog into the original image
        fog_blend_factor = random.uniform(0.5, 1.0)  # Random blend factor for variety
        img = Image.blend(img, fog, fog_blend_factor)
        return img

    def __call__(self, img, target):
        # Apply the same random rotation
        angle = random.uniform(-10, 10)  # Generate random angle
        img = Fv.rotate(img, angle)  
        target = Fv.rotate(target, angle, interpolation=Fv.InterpolationMode.NEAREST)  
        # Apply horizontal flip manually (for label consistency)
        if torch.rand(1) < 0.5:
            img = Fv.hflip(img)
            target = Fv.hflip(target)
        # Apply fog effect on the image
        img = self.add_fog(img)
        img = self.image_transform(img)
        target = self.label_transform(target)
        target = target.to(torch.long)  # Ensure labels are integers
        return img, target

# Load the Cityscapes dataset (ensure you have the correct path to the dataset)
transform = CustomTransform()
cityscapes_dataset = Cityscapes(root='./data', split='train', mode='fine', target_type='semantic', transform=transform.image_transform)

# To visualize the transformation
def visualize_transform(dataset, index=0):
    # Get an image and target from the dataset
    img, target = dataset[index]
    
    # Convert tensor image back to PIL for visualization
    img = img.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    img = img.numpy()  # Convert tensor to numpy for visualization
    img = (img * 0.229 + 0.485) * 255  # Unnormalize the image
    img = img.clip(0, 255).astype('uint8')  # Clip values to valid range
    
    target = target.numpy()  # Get the target as numpy array (labels)
    
    # Plot the original and transformed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Transformed Image")
    axes[0].axis('off')
    
    # Show the target (label) image
    axes[1].imshow(target, cmap='jet')  # Display labels with a jet colormap for better visualization
    axes[1].set_title("Transformed Label")
    axes[1].axis('off')

    plt.show()

# Visualize the transformation on the first image in the dataset
visualize_transform(cityscapes_dataset, index=0)
