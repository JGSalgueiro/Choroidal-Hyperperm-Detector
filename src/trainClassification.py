import os
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import torch.nn.functional as F

classes = ["No", "Yes"]

class ImageClassifier(pl.LightningModule):
    def __init__(self, data_dir, batch_size=4, learning_rate=1e-3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = 2 

        # Build the ResNet-50 model architecture as during training
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
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

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

def main():
    data_dir = "DataSets/classification/Data"
    model = ImageClassifier(data_dir)
    
    #trainer = pl.Trainer(max_epochs=300, accelerator='gpu', devices=1)
    #trainer.fit(model)
    
    # trainer.fit(model)
    save_path = "saved_models/classifier.pth"
    #torch.save(model.state_dict(), save_path

    loaded_model = ImageClassifier(data_dir)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval() 

    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    to_classify_dir = "toClassify"
    for filename in os.listdir(to_classify_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(to_classify_dir, filename)
            image = Image.open(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = inference_transform(image)  
            image = image.unsqueeze(0)  

            with torch.no_grad():
                output = loaded_model(image)
                probabilities = F.softmax(output, dim=1) 
                predicted_class = classes[torch.argmax(probabilities, dim=1).item()]
                print(f"Probabilities: {probabilities}") 

            print(f"Image: {filename}, Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
