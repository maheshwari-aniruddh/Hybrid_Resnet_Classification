import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature, io
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# Parameters
IMAGE_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_NUM_POINTS = 24
LBP_RADIUS = 8


# Cropping function
def crop_image_based_on_threshold(grayscale_image):
    threshold = 0.5
    binary_mask = grayscale_image > threshold
    thresholded_image = np.where(binary_mask, 255, 0).astype(np.uint8)

    rows_to_keep = ~np.all(thresholded_image == 0, axis=1)
    cols_to_keep = ~np.all(thresholded_image == 0, axis=0)

    row_indices = np.where(rows_to_keep)[0]
    col_indices = np.where(cols_to_keep)[0]

    if row_indices.size > 0 and col_indices.size > 0:
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        cropped_image = grayscale_image[min_row:max_row, min_col:max_col]
        return cropped_image
    else:
        return grayscale_image


# Function to load and preprocess a new image
def preprocess_new_image(image_path, size=IMAGE_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Crop the image
    cropped_img = crop_image_based_on_threshold(img)

    # Resize the cropped image
    img_resized = cv2.resize(cropped_img, size)

    # Convert grayscale to a 3-channel image
    img_resized_3d = np.stack((img_resized,) * 3, axis=-1)

    return img_resized_3d


# Function to extract HOG features from an image
def extract_hog_features(image):
    gray_image = image[:, :, 0]  # Use the first channel (grayscale) for HOG
    features = feature.hog(gray_image, pixels_per_cell=HOG_PIXELS_PER_CELL, cells_per_block=HOG_CELLS_PER_BLOCK,
                           visualize=False)
    return features


# Function to extract LBP features from an image
def extract_lbp_features(image, num_points=LBP_NUM_POINTS, radius=LBP_RADIUS):
    gray_image = image[:, :, 0]  # Use the first channel (grayscale) for LBP
    lbp = feature.local_binary_pattern(gray_image, num_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path,
                       custom_objects={"ResNet50": ResNet50})  # Ensure custom objects are loaded correctly if used
    return model


# Predict the probability of having a tumor
def predict_tumor(model, image_path):
    # Preprocess the image
    img = preprocess_new_image(image_path)

    # Extract features
    hog_features = extract_hog_features(img)
    lbp_features = extract_lbp_features(img)

    # Normalize the image
    img_normalized = img.astype('float32') / 255.0

    # Make prediction
    prediction = model.predict([np.expand_dims(img_normalized, axis=0), np.expand_dims(hog_features, axis=0),
                                np.expand_dims(lbp_features, axis=0)])

    return prediction[0][0] 


# Example usage
model_path = 'brain_tumor_detection_model.h5'  # Replace with your actual model path
image_path = 'Y13.jpg'  # Replace with the path to the image you want to predict

import os


# Function to process a folder of images and save probabilities in a list
import os


# Function to process a folder of images and save probabilities in an array
def process_image_folder(model, folder_path):
    probabilities = []  # List to store probabilities
    filenames = []  # List to store corresponding filenames

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                # Predict the probability of having a tumor for the current image
                probability = predict_tumor(model, image_path)
                probabilities.append(probability)
                filenames.append(filename)

                # Print result for each image (optional)
                print(f"Image: {filename} - Probability of having a tumor: {probability * 100:.2f}%")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return filenames, probabilities


# Example usage
folder_path = '/Users/aniruddhmodi/PycharmProjects/Research/archive/brain_tumor_dataset/no'  # Replace with the path to the folder containing the images

model = load_trained_model(model_path)  # Load your trained model
filenames, probabilities = process_image_folder(model, folder_path)

# Print or save the results
print("Image filenames and their corresponding probabilities:")
for filename, prob in zip(filenames, probabilities):
    print(prob*100)

# You can also save the results to a file or further process them
