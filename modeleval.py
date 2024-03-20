import torch
import random
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from dataset import FireSegmentationDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Defines a class for performing Grad Cam visualization
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    # Saves gradients
    def save_gradients(self, grad):
        self.gradients = grad

    # Saves activations
    def save_activations(self, act):
        self.activations = act

    # Captures activations with forward hook and gradients with backwards hook
    def register_hooks(self):
        self.target_layer.register_forward_hook(lambda module, input, output: self.save_activations(output))
        self.target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: self.save_gradients(grad_out[0]))

    # Computes gradcam using input tensor
    def compute_gradcam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, dict):
            output = output['out'] 
        
        if output.size(1) == 1:
            one_hot_output = torch.ones_like(output)
            one_hot_output.requires_grad_(True)
        else:
            raise NotImplementedError("GradCam is not implemented for multi-class segmentation in this context.")

        self.model.zero_grad()
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        guided_gradients = self.gradients.data[0]
        target = self.activations.data[0]
        weights = guided_gradients.mean(dim=(1, 2), keepdim=True)
        grad_cam = torch.mul(target, weights).sum(dim=0).relu()
        
        return grad_cam

# Contains functions to evaluate the saved model
class ModelEvaluator:

    # Initializes the class with transformed datasets and preprocess transforms for functions
    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model().to(self.device)
        self.load_model(model_path)
        self.dataset = dataset
        self.target_layer = self.model.encoder.layer4[-1]

        self.image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        self.dataset = FireSegmentationDataset.load_datasets(
            dataset,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform
        )
    
    # Initializes model before weights are attached
    def initialize_model(self):
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        return model

    # Loads model from path onto object
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # Evaluated model on testing set (loss and IoU)
    def evaluate_model_on_test(self, criterion):
        test_loader = DataLoader(self.dataset['test'], batch_size=32, shuffle=False, num_workers=4)
        total_loss, total_iou = 0.0, 0.0

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
    def grad_cam_visualization(self, image_path):
        original_image = Image.open(image_path).convert("RGB")
        input_tensor = self.image_transform(original_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        grad_cam = GradCam(self.model, self.target_layer)
        cam_output = grad_cam.compute_gradcam(input_tensor)
        
        cam_output -= cam_output.min()
        cam_output /= cam_output.max()
        original_image_size = original_image.size
        cam_output_resized = F.interpolate(cam_output.unsqueeze(0).unsqueeze(0), size=original_image_size, mode='bilinear', align_corners=False).squeeze()
        cam_output_resized = cam_output_resized.cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')  

        ax[1].imshow(original_image)
        ax[1].imshow(cam_output_resized, cmap='jet', alpha=0.5)
        ax[1].set_title('Grad-CAM')
        ax[1].axis('off') 

        plt.tight_layout()
        plt.savefig('Visuals/gradcam.jpg')
        plt.show()
    

image_dir = 'dataset/Image'
mask_dir = 'dataset/Segmentation_Mask'
model_path = 'best_model.pth'
dataset_splits_path = 'dataset/dataset_splits.pkl'

evaluator = ModelEvaluator(model_path, dataset_splits_path)
criterion = torch.nn.BCEWithLogitsLoss()

evaluator.evaluate_model_on_test(criterion)

evaluator.grad_cam_visualization('dataset/Image/Fire/Img_19431.jpg')
