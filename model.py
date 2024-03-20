import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import FireSegmentationDataset  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import numpy as np

# Class for early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    # Method to call for checking and applying early stopping
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Trainer class for U-Net model
class UnetTrainer:

    # Initialization with dataset paths and model setup
    def __init__(self, dataset_splits_path, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        self.model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        mask_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])

        datasets = FireSegmentationDataset.load_datasets(
            dataset_splits_path,
            image_transform=image_transform,
            mask_transform=mask_transform
        )

        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.test_dataset = datasets['test']

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.3)
        self.early_stopping = EarlyStopping(patience=5, verbose=True)

    # Training method
    def train(self, epochs, criterion):
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=4)

        best_iou = 0.0
        train_losses, val_losses, train_ious, val_ious = [], [], [], []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval()
            with torch.no_grad():
                total_train_iou = 0.0
                for images, masks in train_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    predicted_masks = torch.sigmoid(outputs) > 0.5
                    total_train_iou += self.calculate_iou(predicted_masks, masks)
                avg_train_iou = total_train_iou / len(train_loader)
            train_ious.append(avg_train_iou)

            val_loss, val_iou = self.evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            val_ious.append(val_iou)

            print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train IoU: {avg_train_iou}, Validation Loss: {val_loss}, Validation IoU: {val_iou}")

            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), 'best_model.pth')
                
        self.plot_metrics(train_losses, val_losses, train_ious, val_ious)

    # Method to calculate IoU
    def calculate_iou(self, outputs, masks):
        outputs = outputs.float()
        masks = masks.float()
        intersection = (outputs * masks).sum((1, 2))
        union = (outputs + masks).clamp(0, 1).sum((1, 2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

    # Method for evaluating model performance on validation or test dataset
    def evaluate(self, data_loader, criterion):
        total_loss, ious = 0.0, []
        for images, masks in data_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            predicted_masks = torch.sigmoid(outputs) > 0.5
            iou = self.calculate_iou(predicted_masks, masks)
            ious.append(iou)
        avg_loss = total_loss / len(data_loader)
        avg_iou = sum(ious) / len(ious)
        return avg_loss, avg_iou

    # Method to plot and save training metrics like loss and IoU
    def plot_metrics(self, train_losses, val_losses, val_ious, train_ious_epochs):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, val_losses, label='Validation Loss', color='green')
        plt.plot(epochs, train_losses, label='Training Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, val_ious, label='Validation IoU', color='green')
        plt.plot(epochs, train_ious_epochs, label='Training IoU', color='red')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()


trainer = UnetTrainer(dataset_splits_path="dataset/dataset_splits.pkl")
criterion = nn.BCEWithLogitsLoss() 

trainer.train(epochs=10, criterion=criterion)