import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models

from customDataset import CustomDataset2
from models import get_model, UNetPPDecoder


img_folder = "DataSets/classification/img"
mask_folder = "DataSets/classification/masked"
val_img_folder = "DataSets/classification/val"
val_mask_folder = "DataSets/classification/val_masks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dice
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def apply_custom_threshold(self, mask, threshold):
        return torch.sigmoid(mask - threshold)

    def forward(self, outputs, masks):
        smooth = 1e-6

        # Calculate the threshold
        threshold = (masks.min() + masks.max()) / 2

        # Apply custom thresholding to both outputs and masks
        thresholded_outputs = self.apply_custom_threshold(outputs, threshold)
        thresholded_masks = self.apply_custom_threshold(masks, threshold)

        thresholded_outputs = thresholded_outputs.float().requires_grad_()
        thresholded_masks = thresholded_masks.float().requires_grad_()

        intersection = torch.sum(thresholded_outputs * thresholded_masks)
        union = torch.sum(thresholded_outputs + thresholded_masks) + smooth
        dice_loss = 1 - (2.0 * intersection + smooth) / (union - intersection)
        return dice_loss


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.unet = get_model('seuneter_multi').to(device)
        self.train_loss_values = []
        self.val_loss_values = []
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.bce_loss_fn(outputs, masks)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def check_convergence(self, patience=5):
        if len(self.val_loss_values) < patience:
            return False
        else:
            for i in range(1, patience + 1):
                if self.val_loss_values[-i] < self.val_loss_values[-(i + 1)]:
                    return False
            return True

# doesnt work ffs
class PreTrained(pl.LightningModule):
    def __init__(self, pretrained_encoder=True):
        super(PreTrained, self).__init__()

        if pretrained_encoder:
            self.encoder = models.resnet50(pretrained=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder = get_model('drunet').to(device)

        self.encoder_channels = [64, 128, 256, 512]  # Encoder output channels
        self.decoder_channels = [512, 256, 128, 64]  # Decoder input channels
        self.decoder = UNetPPDecoder(self.encoder_channels, self.decoder_channels).to(device)

        self.train_loss_values = []
        self.val_loss_values = []

        self.dice_loss_fn = DiceLoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        encoded_features = self.encoder(x)
        decoded_output = self.decoder(encoded_features[-1], encoded_features[:-1])
        return decoded_output

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.bce_loss_fn(outputs, masks)

        self.train_loss_values.append(loss.item())
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        val_losses = self.trainer.callback_metrics.get("val_loss")
        if val_losses is not None:
            if isinstance(val_losses, torch.Tensor):
                val_losses = [val_losses]
            self.val_loss_values.extend([l.item() for l in val_losses])

        avg_val_loss = torch.tensor(self    .val_loss_values).abs().mean()
        self.log("val_loss", avg_val_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        dice_loss = self.dice_loss_fn(outputs, masks)
        bce_loss = self.bce_loss_fn(outputs, masks)

        loss = bce_loss

        print(dice_loss.item())

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def check_convergence(self, patience=5):
        if len(self.val_loss_values) < patience:
            return False
        else:
            for i in range(1, patience + 1):
                if self.val_loss_values[-i] < self.val_loss_values[-(i + 1)]:
                    return False
            return True


if __name__ == '__main__':
    transform = ToTensor()
    #val_dataset = CustomDataset(val_img_folder, val_mask_folder, transform=transform)
    #val_loader = DataLoader(val_dataset, batch_size=11, shuffle=False, num_workers=4)
    dataset = CustomDataset2(img_folder, mask_folder, transform=transform)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    model = Model()
    #early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    trainer = pl.Trainer(max_epochs=500, accelerator='gpu', devices=1)
    
    
    trainer.fit(model, train_loader)  # Add val_loader for validation
    save_path = "saved_models/model.pth"
    torch.save(model.state_dict(), save_path)

    num_train_steps = len(model.train_loss_values)
    num_val_epochs = len(model.val_loss_values)
    stretched_val_loss_values = np.repeat(model.val_loss_values, num_train_steps // num_val_epochs)

    plt.figure()
    plt.plot(model.train_loss_values, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

    if model.check_convergence():
        print("Network has converged. Stopping training.")
    else:
        print("Network has not converged. Consider increasing.")

