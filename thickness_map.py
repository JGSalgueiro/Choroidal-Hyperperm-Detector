import os
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from customDataset import CustomDatasetForMaps
from models import get_model
from PIL import Image
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np


img_folder = "DataSets/ChoroidSegmentation/thickness_map"
dummy_mask = "DataSets/ChoroidSegmentation/thickness_map"
mask_folder = "DataSets/ChoroidSegmentation/thickness_masks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetModel(pl.LightningModule):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.unet = get_model('seuneter').to(device)

        self.train_loss_values = []
        self.val_loss_values = []

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_loss_values.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_loss_values.append(loss.item())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

def measure_thickness(image_path):
    image = Image.open(image_path).convert('L')
    #image = image.resize((684, 684))
    image = image.resize((500, 500))
    image_array = np.array(image)
    thickness = np.sum(image_array == 255, axis=0)

    return thickness

def create_thickness_map(folder_path):
    thickness_matrix = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            thickness = measure_thickness(image_path)
            thickness_matrix.append(thickness)

    if len(thickness_matrix) == 0:
        raise ValueError("No valid images found in the folder.")

    thickness_map = np.array(thickness_matrix)

    return thickness_map

def apply_custom_threshold(mask, threshold):
    return torch.where(mask <= threshold, torch.tensor(0, device=mask.device), torch.tensor(255, device=mask.device))

def segment_images(model, img_folder, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)

    transform = ToTensor()
    dataset = CustomDatasetForMaps(img_folder, img_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Number of images: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")

    if len(data_loader) == 0:
        print("Data loader is empty.")

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (image, _, image_name) in enumerate(data_loader):
            print(f"Processing image {i + 1}/{len(data_loader)}")
            image = image.to(device)
            mask = model(image)
            mask = torch.sigmoid(mask)
            threshold = (mask.min() + mask.max()) / 2
            mask = apply_custom_threshold(mask, threshold)
            image_name = image_name[0].split(".")[0] + ".png"
            mask_path = os.path.join(mask_folder, image_name)
            mask_image = Image.fromarray(mask[0, 0].byte().cpu().numpy(), mode="L")
            mask_image.save(mask_path)

            print(f"Segmented mask saved: {mask_path}")

    print("Segmentation complete.")

if __name__ == '__main__':
    model = UNetModel()
    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    segment_images(model,img_folder,mask_folder )

