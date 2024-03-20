# Image Segmentation Task - Fire Segmentation

## Code

* dataset.py contains code to prepare and load data onto splits
* datasetvis.py contains code to find information and examples of dataset
* model.py contains code to set up and train UNet model on prepared data
* modeleval.py contains code to evaluate saved model
* Requires pytorch installations (torch, torchvision)

## Data

* I utilized the Fire Image Segmentation dataset from Kaggle:
    * https://www.kaggle.com/datasets/diversisai/fire-segmentation-image-dataset
    * I extracted it as a folder called dataset
    * For reproduction of code, dataset_split.pkl should be placed inside of dataset folder with Image and Segmentation_Mask folders


