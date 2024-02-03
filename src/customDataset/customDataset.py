import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor


class CustomDatasetForMaps(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(img_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        
    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx % len(self.image_files)])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx % len(self.mask_files)])
        
        image = Image.open(img_path).convert("L")  # Convert to grayscale 
        mask = Image.open(mask_path).convert("L")
        
        # Resize images to 256x256 
        resize_transform = Resize((256, 256))
        image = resize_transform(image)
        mask = resize_transform(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        image_name = self.image_files[idx % len(self.image_files)]  # Only for val XD
        
        return image, mask, image_name


class CustomDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(img_folder))
        self.mask_files = sorted(os.listdir(mask_folder))

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx % len(self.image_files)])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx % len(self.mask_files)])

        image = Image.open(img_path).convert("L")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")

        # Resize images to 256x256
        resize_transform = Resize((256, 256))
        image = resize_transform(image)
        mask = resize_transform(mask)

        # Apply transform to tensor
        transform_to_tensor = ToTensor()
        image = transform_to_tensor(image)
        mask = transform_to_tensor(mask)

        image_name = self.image_files[idx % len(self.image_files)]  # Only for val XD

        return image, mask

class CustomDataset2(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(img_folder))
        self.mask_files = sorted(os.listdir(mask_folder))

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx % len(self.image_files)])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx % len(self.mask_files)])

        image = Image.open(img_path).convert("RGB")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")

        # Resize images to 256x256
        resize_transform = Resize((256, 256))
        image = resize_transform(image)
        mask = resize_transform(mask)

        # Apply transform to tensor
        transform_to_tensor = ToTensor()
        image = transform_to_tensor(image)
        mask = transform_to_tensor(mask)

        image_name = self.image_files[idx % len(self.image_files)]  # Only for val XD

        return image, mask