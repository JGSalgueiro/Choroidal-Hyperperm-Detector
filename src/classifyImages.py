
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class ImageClassifier(pl.LightningModule):
    def __init__(self, data_dir, batch_size=4, learning_rate=1e-3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = 2 

        # Build the same model architecture as during training
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        model = torchvision.models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, self.num_classes)
        return model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.ImageFolder(root=self.data_dir, transform=transform)

        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Define the transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
model = ImageClassifier(data_dir="path_to_your_data_directory")
model.load_state_dict(torch.load("saved_models/classifier.pth"))
model.eval()

# Define a function to classify an image
def classify_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Path to the folder containing images to classify
to_classify_dir = "toClassify"

# Get a list of image file paths in the "toClassify" folder
image_paths = [os.path.join(to_classify_dir, filename) for filename in os.listdir(to_classify_dir)]

# Classify each image and print the result
for image_path in image_paths:
    predicted_class = classify_image(image_path, model)
    if predicted_class == 0:
        result = "NO"
    else:
        result = "YES"
    print(f"Image '{image_path}' is classified as {result}")
