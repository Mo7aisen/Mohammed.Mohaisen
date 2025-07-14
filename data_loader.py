import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Dataset Paths (as provided by you) ---
# This centralized configuration makes it easy to manage your dataset locations.
DATASET_PATHS = {
    "montgomery": {
        "images": "/home/mohaisen_mohammed/Datasets/MontgomeryDataset/CXR_png/",
        "left_masks": "/home/mohaisen_mohammed/Datasets/MontgomeryDataset/ManualMask/leftMask/",
        "right_masks": "/home/mohaisen_mohammed/Datasets/MontgomeryDataset/ManualMask/rightMask/",
    },
    "jsrt": {
        "images": "/home/mohaisen_mohammed/Datasets/JSRT/images/",
        "left_lung_masks": "/home/mohaisen_mohammed/Datasets/JSRT/masks_png/left_lung/",
        "right_lung_masks": "/home/mohaisen_mohammed/Datasets/JSRT/masks_png/right_lung/",
    }
}


class LungDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the JSRT and Montgomery datasets.
    It handles the specific file structures of both datasets, including combining
    left and right lung masks into a single ground truth mask.
    """

    def __init__(self, dataset_name, transform=None, data_fraction=1.0):
        """
        Args:
            dataset_name (str): 'jsrt' or 'montgomery'.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_fraction (float): Fraction of the data to use (for training smaller models).
        """
        self.dataset_name = dataset_name.lower()
        self.transform = transform

        if self.dataset_name not in DATASET_PATHS:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Use 'jsrt' or 'montgomery'.")

        self.image_dir = DATASET_PATHS[self.dataset_name]["images"]
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        # Use a fraction of the dataset if specified
        if data_fraction < 1.0:
            num_samples = int(len(self.image_files) * data_fraction)
            self.image_files = self.image_files[:num_samples]
            print(f"Using {data_fraction * 100:.2f}% of the data: {num_samples} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)

        image = Image.open(img_path).convert("L")

        # --- Mask Loading Logic ---
        # This logic is specific to how your datasets are structured.
        if self.dataset_name == 'montgomery':
            left_mask_path = os.path.join(DATASET_PATHS['montgomery']['left_masks'], base_name)
            right_mask_path = os.path.join(DATASET_PATHS['montgomery']['right_masks'], base_name)

            left_mask = np.array(Image.open(left_mask_path).convert("L"))
            right_mask = np.array(Image.open(right_mask_path).convert("L"))

            mask_np = np.maximum(left_mask, right_mask)
            mask = Image.fromarray(mask_np)

        elif self.dataset_name == 'jsrt':
            # JSRT masks for different parts are in separate folders. We combine them.
            left_mask_path = os.path.join(DATASET_PATHS['jsrt']['left_lung_masks'], base_name)
            right_mask_path = os.path.join(DATASET_PATHS['jsrt']['right_lung_masks'], base_name)

            left_mask = np.array(Image.open(left_mask_path).convert("L")) if os.path.exists(
                left_mask_path) else np.zeros(image.size[::-1], dtype=np.uint8)
            right_mask = np.array(Image.open(right_mask_path).convert("L")) if os.path.exists(
                right_mask_path) else np.zeros(image.size[::-1], dtype=np.uint8)

            mask_np = np.maximum(left_mask, right_mask)
            mask = Image.fromarray(mask_np)

        else:
            # Fallback for unknown datasets
            mask = Image.new("L", image.size, 0)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ResizeAndToTensor(object):
    """
    A transform to resize images and masks and convert them to PyTorch tensors.
    This ensures all data fed to the model has a consistent size.
    """

    def __init__(self, size=(256, 256)):
        self.size = size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        image_tensor = self.to_tensor(image)
        # Normalize the image tensor
        image_tensor = self.normalize(image_tensor)

        mask_tensor = self.to_tensor(mask)
        # Ensure mask is binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()

        return {'image': image_tensor, 'mask': mask_tensor}


if __name__ == '__main__':
    # --- Example of how to use the DataLoader ---
    # This part runs only if you execute `python data_loader.py` directly.
    # It's useful for testing and debugging your data loading process.

    print("Testing Montgomery DataLoader...")
    try:
        montgomery_dataset = LungDataset(dataset_name='montgomery', transform=ResizeAndToTensor())
        montgomery_loader = DataLoader(montgomery_dataset, batch_size=4, shuffle=True)
        sample_batch = next(iter(montgomery_loader))
        print(f"Montgomery - Image batch shape: {sample_batch['image'].shape}")
        print(f"Montgomery - Mask batch shape: {sample_batch['mask'].shape}")
        print("Montgomery DataLoader test successful.")
    except Exception as e:
        print(f"Error testing Montgomery DataLoader: {e}")
        print("Please ensure your dataset paths are correct and you have access to the server.")

    print("\nTesting JSRT DataLoader...")
    try:
        jsrt_dataset = LungDataset(dataset_name='jsrt', transform=ResizeAndToTensor())
        jsrt_loader = DataLoader(jsrt_dataset, batch_size=4, shuffle=True)
        sample_batch = next(iter(jsrt_loader))
        print(f"JSRT - Image batch shape: {sample_batch['image'].shape}")
        print(f"JSRT - Mask batch shape: {sample_batch['mask'].shape}")
        print("JSRT DataLoader test successful.")
    except Exception as e:
        print(f"Error testing JSRT DataLoader: {e}")
        print("Please ensure your dataset paths are correct and you have access to the server.")
