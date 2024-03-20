import torch
import random
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from dataset import FireSegmentationDataset
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Defines a target for Grad-CAM that focuses on the pixels belonging to a specific class.
class SemanticSegmentationTarget:
    def __init__(self, category_mask):
        self.category_mask = category_mask

    def __call__(self, model_output):
        return (model_output.squeeze(1) * self.category_mask).sum()

# Contains functions to evaluate the saved model
class ModelEvaluator:

    # Initializes the class with transformed datasets and preprocess transforms for functions
    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model().to(self.device)
        self.load_model(model_path)
        self.dataset = dataset
        conv_layers = self.find_last_conv_layer(self.model)
        self.target_layers = [conv_layers[0], conv_layers[-1]]
        
        self.preprocess = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        self.dataset = FireSegmentationDataset.load_datasets(
            dataset,
            image_transform=image_transform,
            mask_transform=mask_transform
        )
    
    # Initializes model before weights are attached
    def initialize_model(self):
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        return model

    # Loads model from path onto object
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # Finds last layer for use in GradCam
    def find_last_conv_layer(self, model):
        conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.conv.Conv2d)]
        return conv_layers

    # Evaluated model on testing set (loss and IoU)
    def evaluate_model_on_test(self, criterion):
        test_loader = DataLoader(self.dataset['test'], batch_size=32, shuffle=False, num_workers=4)
        total_loss, total_iou, all_f1_scores = 0.0, 0.0, []

        for images, masks in test_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            predicted_probs = torch.sigmoid(outputs)
            predicted_masks = (predicted_probs > 0.5).float()
            
            total_iou += self.calculate_iou(predicted_masks, masks)

        avg_loss = total_loss / len(test_loader)
        avg_iou = total_iou / len(test_loader)

        print(f"Test Loss: {avg_loss}, Test IoU: {avg_iou}")
        return avg_loss, avg_iou

    # Calculates Intersection over Union value
    def calculate_iou(self, preds, labels):

        preds_bool = preds > 0.5  
        labels_bool = labels > 0.5 
        
        intersection = (preds_bool & labels_bool).float().sum((1, 2))
        union = (preds_bool | labels_bool).float().sum((1, 2)) 
        
        iou = (intersection + 1e-6) / (union + 1e-6) 
        return torch.mean(iou).item()

    # Creates Grad Cam Visualization
    def grad_cam_visualization(self, image_path, target_category):
        input_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            model_output = self.model(input_tensor)
            predicted_mask = torch.sigmoid(model_output) > 0.5
            predicted_mask = predicted_mask.squeeze().cpu().numpy()

        category_mask = (predicted_mask == target_category).astype(float)
        target = SemanticSegmentationTarget(torch.tensor(category_mask).to(self.device))

        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0, :]
        
        original_image = Image.open(image_path).convert('RGB')
        original_image_np = np.array(original_image) / 255.0

        heatmap_resized = np.array(to_pil_image(grayscale_cam).resize(original_image.size, Image.BICUBIC))
        visualization = show_cam_on_image(original_image_np, heatmap_resized, use_rgb=True)
        
        plt.imshow(visualization)
        plt.savefig('gradcam.png')
    
    # Processes image for use in grad cam visualization
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0).to(self.device)

image_dir = 'dataset/Image'
mask_dir = 'dataset/Segmentation_Mask'
model_path = 'best_model.pth'
dataset_splits_path = 'dataset/dataset_splits.pkl'

evaluator = ModelEvaluator(model_path, dataset_splits_path)
criterion = torch.nn.BCEWithLogitsLoss()

test_loss, test_iou = evaluator.evaluate_model_on_test(criterion)

import pickle
import random

with open('dataset/dataset_splits.pkl', 'rb') as file:
    dataset_splits = pickle.load(file)
test_image_paths = dataset_splits['test']
random_test_image_path = random.choice(test_image_paths)
print(random_test_image_path)

evaluator.grad_cam_visualization(random_test_image_path[0], target_category=1)