# Technologies Used

Python:Python is the main programming language for writing the CNN model, training, and testing the denoising system.

MATLAB:MATLAB can be used for initial noise experiments and filter visualization before moving to deep learning implementation.

CNN:The CNN learns how to remove noise by comparing noisy and clean images and reconstructing the denoised version.

Opencv:OpenCV is used for preprocessing (adding noise, resizing, normalization) and post-processing (displaying results).

TENSORFLOW:TensorFlow is the core backend where CNN is defined and trained.

Keras:Keras is used to build and train the CNN model for image denoising.

ML:CNN is a part of supervised ML, trained on noisy vs clean images.

google colab:Used for training your CNN model efficiently with GPU support.



# Google Colab
from tensorflow import keras

import numpy as np

import cv2

import os

import matplotlib.pyplot as plt

from google.colab import drive

# def plot_training_history(self, history):
plt.figure(figsize=(12, 6))

plt.plot(history.history['mean_absolute_error'], label='Train MAE', linewidth=2)

plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE', linewidth=2)

plt.xlabel("Epochs", fontsize=14)

plt.ylabel("Mean Absolute Error", fontsize=14)

plt.legend(fontsize=14)

plt.title("Training History", fontsize=16)

plt.grid(True)

plt.show()

# Define paths
dataset_path = '/content/drive/MyDrive/train'

save_model_path = '/content/drive/MyDrive/saved_model.h5'

# Screenshots

Noise to Denoise 

<img width="1286" height="729" alt="Screenshot 2025-11-14 183353" src="https://github.com/user-attachments/assets/d13c0395-d44f-45b1-89b3-f70a3908c555" />

<img width="1131" height="718" alt="Screenshot 2025-11-14 183433" src="https://github.com/user-attachments/assets/faee0a67-aa0f-4888-a1bd-6a30c0c0d5de" />

# Training History

<img width="1033" height="738" alt="Screenshot 2025-11-14 183453" src="https://github.com/user-attachments/assets/349b687d-d393-4e9d-ba2f-a83bf3d3d0a9" />


# Just follow me and Star⭐ my repository

