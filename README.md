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
