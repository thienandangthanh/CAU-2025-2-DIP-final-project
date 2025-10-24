import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LowLightDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        """
        Initializes the dataset.

        Args:
            data_dir (str): The directory containing the low-light images.
            image_size (int): The size to which images will be resized.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Get a list of all image file names in the directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Define the image transformations
        self.transform = transforms.Compose([
            # Resize the image to a square of size image_size x image_size
            transforms.Resize((self.image_size, self.image_size)),
            # Convert the image to a PyTorch tensor (scales pixel values to [0, 1])
            transforms.ToTensor()
        ])

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The transformed low-light image.
        """
        # Construct the full path to the image file
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Open the image using Pillow and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply the defined transformations
        low_light_image = self.transform(image)
        
        return low_light_image

# --- Simple test to verify implementation ---
if __name__ == '__main__':
    # Before running this, make sure you have a folder `data/train`
    # and at least one image inside it.
    
    # Create a dummy folder and image for testing purposes
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    try:
        Image.new('RGB', (600, 400), color = 'red').save('data/train/dummy_image.jpg')
        
        # Path to your training data
        train_dir = 'data/train'
        
        # Create an instance of the dataset
        train_dataset = LowLightDataset(data_dir=train_dir)
        
        # Check the length of the dataset
        print(f"Found {len(train_dataset)} images in {train_dir}")
        
        # Get the first item from the dataset
        first_image = train_dataset[0]
        
        # Check the shape and type of the returned tensor
        print(f"Shape of the first image tensor: {first_image.shape}")
        print(f"Data type of the tensor: {first_image.dtype}")
        print(f"Min value: {first_image.min()}, Max value: {first_image.max()}")
        print("\nDataset implemented successfully!")
        
        # Clean up the dummy file
        os.remove('data/train/dummy_image.jpg')
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        print("Please ensure you have a 'data/train' folder with at least one image.")