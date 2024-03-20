import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# A class to visualize datasets and save images and masks.
class DatasetVisualizer:

    # Initialize with the directory of images and masks.
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    # Display and save a specified number of random images from a category.
    def display_and_save_random_images(self, category, save_path='.', n_images=3):
        category_path = os.path.join(self.image_dir, category)
        chosen_files = random.sample(os.listdir(category_path), n_images)
        fig, axs = plt.subplots(1, n_images, figsize=(n_images * 5, 5))

        for i, file_name in enumerate(chosen_files):
            img_path = os.path.join(category_path, file_name)
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].set_title(f'{category}: {file_name}')
            axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{category}_random_images.png'))
        plt.close()

    # Display and save masks corresponding to a list of image names.
    def display_and_save_masks(self, image_names, save_path='.'):
        n_images = len(image_names)
        fig, axs = plt.subplots(1, n_images, figsize=(n_images * 5, 5))

        for i, image_name in enumerate(image_names):
            mask_path = os.path.join(self.mask_dir, image_name)
            mask = Image.open(mask_path).convert("L")
            axs[i].imshow(mask, cmap='gray')
            axs[i].set_title(f'Segmentation Mask: {image_name}')
            axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'segmentation_masks.png'))
        plt.close()

    # Display and save a comparison of images and their segmentation masks.
    def display_and_save_comparison(self, category, image_names, save_path='.'):
        n_images = len(image_names)
        fig, axs = plt.subplots(n_images, 2, figsize=(10, n_images * 5))

        for i, image_name in enumerate(image_names):
            img_path = os.path.join(self.image_dir, category, image_name)
            img = Image.open(img_path)
            mask_path = os.path.join(self.mask_dir, image_name)
            mask = Image.open(mask_path).convert("L")

            axs[i, 0].imshow(img)
            axs[i, 0].axis('off')
            axs[i, 1].imshow(mask, cmap='gray')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{category}_comparison.png'))
        plt.close()

    # Plot and save a bar chart of the distribution of images across categories.
    def plot_class_distribution(self, save_path='.'):
        categories = ['Fire', 'Not_Fire']
        counts = [len(os.listdir(os.path.join(self.image_dir, cat))) for cat in categories]

        plt.bar(categories, counts, color=['red', 'blue'])
        plt.title('Number of Images in Each Category')
        plt.xlabel('Category')
        plt.ylabel('Number of Images')
        plt.xticks(categories)

        plt.savefig(os.path.join(save_path, 'class_distribution.png'))
        plt.close()

image_dir = 'dataset/Image'
mask_dir = 'dataset/Segmentation_Mask/Fire'

visualizer = DatasetVisualizer(image_dir, mask_dir)

visualizer.display_and_save_random_images('Fire', save_path='Visuals', n_images=3)
visualizer.display_and_save_random_images('Not_Fire', save_path='Visuals', n_images=3)

image_names = ['Img_0.jpg', 'Img_10500.jpg', 'Img_23000.jpg']
visualizer.display_and_save_masks(image_names, save_path='Visuals')

visualizer.display_and_save_comparison('Fire', image_names, save_path='Visuals')

visualizer.plot_class_distribution(save_path='Visuals')