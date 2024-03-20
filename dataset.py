import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# A dataset class for fire segmentation that includes methods for data preparation and loading.
class FireSegmentationDataset(Dataset):

    # Initialize with paths to images and masks and optional transforms.
    def __init__(self, dataset_info, image_transform=None, mask_transform=None):
        self.dataset_info = dataset_info
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    # Return the length of the dataset.
    def __len__(self):
        return len(self.dataset_info)

    # Retrieve an image and its mask at a given index.
    def __getitem__(self, idx):
        img_path, mask_path = self.dataset_info[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path else Image.new('L', image.size, 0)
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

    # Prepare dataset by splitting into training, validation, and test sets and save to a file.
    @staticmethod
    def prepare_data(fire_img_dir, not_fire_img_dir, mask_dir, save_path):
        fire_images = [(os.path.join(fire_img_dir, img), os.path.join(mask_dir, img)) for img in os.listdir(fire_img_dir)]
        not_fire_images = [(os.path.join(not_fire_img_dir, img), None) for img in os.listdir(not_fire_img_dir)]
        all_images = fire_images + not_fire_images

        train_val, test = train_test_split(all_images, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=(2/9), random_state=42)

        with open(save_path, 'wb') as f:
            pickle.dump({'train': train, 'val': val, 'test': test}, f)

    # Load datasets from a saved file and apply optional transforms.
    @staticmethod
    def load_datasets(save_path, image_transform=None, mask_transform=None):
        with open(save_path, 'rb') as f:
            datasets = pickle.load(f)
        return {
            'train': FireSegmentationDataset(datasets['train'], image_transform, mask_transform),
            'val': FireSegmentationDataset(datasets['val'], image_transform, mask_transform),
            'test': FireSegmentationDataset(datasets['test'], image_transform, mask_transform)
        }
    
# FireSegmentationDataset.prepare_data(
#     fire_img_dir="dataset/Image/Fire",
#     not_fire_img_dir="dataset/Image/Not_Fire",
#     mask_dir="dataset/Segmentation_Mask/Fire",
#     save_path="dataset/dataset_splits.pkl"
# )
