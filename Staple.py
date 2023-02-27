# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:27:29 2023

@author: 20192059
"""
import imageio
import SimpleITK as sitk 
import numpy as np
from matplotlib import pyplot as plt

# Load atlas
seg1 = imageio.v2.imread(r"C:\Users\20192059\Documents\Master\CS in MIA\TrainingData\TrainingData\p125\prostaat.mhd")[40,:,:]
seg2 = imageio.v2.imread(r"C:\Users\20192059\Documents\Master\CS in MIA\TrainingData\TrainingData\p120\prostaat.mhd")[40,:,:]
seg3 = imageio.v2.imread(r"C:\Users\20192059\Documents\Master\CS in MIA\TrainingData\TrainingData\p127\prostaat.mhd")[40,:,:]

# Convert to SITK image objects
seg1_sitk = sitk.GetImageFromArray(seg1.astype(np.int16)) 
seg2_sitk = sitk.GetImageFromArray(seg2.astype(np.int16))
seg3_sitk = sitk.GetImageFromArray(seg3.astype(np.int16))
seg_stack = [seg1_sitk, seg2_sitk, seg3_sitk]

# Run STAPLE  algorithm
STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 )
STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)

# Threshold fused image
STAPLE_seg[STAPLE_seg<0.9] = 0    
STAPLE_seg[STAPLE_seg>0.9] = 1

# Display atlas
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(seg1, cmap='gray')
ax[0].set_title('Segmentation 1')
ax[1].imshow(seg2, cmap='gray')
ax[1].set_title('Segmentation 2')
ax[2].imshow(seg3, cmap='gray')
ax[2].set_title('Segmentation 3')
ax[3].imshow(STAPLE_seg, cmap='gray')
ax[3].set_title('Fusion')

plt.show()

