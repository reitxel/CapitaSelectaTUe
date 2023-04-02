# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:08:55 2023

@author: raque
"""

import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import SimpleITK as sitk
import os
from PIL import Image
import torchvision.transforms as transforms



import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# this is my best epoch - what is yours?
BEST_EPOCH = 51
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights" / f"u_net_{BEST_EPOCH}.pth"

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

images = []
path_to_folder = r'C:\Users\raque\OneDrive\Escritorio\UU\capita_selecta_medical_tue\code_teacher\generated\Run2_l1_extra_layers_2_zdim'
# loop through all files in the folder
for filename in os.listdir(path_to_folder):
    if filename.endswith(".mhd"):
        file_path = os.path.join(path_to_folder, filename)
        image = sitk.ReadImage(file_path)
        image = sitk.GetArrayFromImage(image)
        images.append(image)
        #plt.imshow(image, cmap='gray')
        #plt.show()
        
        with torch.no_grad():
            convert_pil = transforms.ToPILImage()
            img = convert_pil(image)
            convert_tensor = transforms.ToTensor()
            img = convert_tensor(img)
            input = img
            output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
            prediction = torch.round(output)
            
            # create folders to save labels and mr with its correct name
            name, extension = os.path.splitext(filename)
            path_to_save = os.path.join(DATA_DIR, name)
            
            if not os.path.exists(os.path.join(DATA_DIR, name)):
                os.makedirs(os.path.join(DATA_DIR, name))
                
            path_to_save_mr = os.path.join(DATA_DIR, name, "mr_bffe" + ".mhd")
            path_to_save_label = os.path.join(DATA_DIR, name, "prostaat" + ".mhd")
            
            mr = sitk.GetImageFromArray(img[0])
            sitk.WriteImage(mr, path_to_save_mr)
    
            label = sitk.GetImageFromArray(prediction[0][0])
            sitk.WriteImage(label, path_to_save_label)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(input[0], cmap="gray")
            ax[0].set_title("Input")
            ax[0].axis("off")

            ax[1].imshow(prediction[0][0])
            ax[1].set_title("Prediction")
            ax[1].axis("off")
            plt.show()
        
        