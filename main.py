# Invariant Feature Detection for Brain Tumor Detection Using Group Theory
# This code demonstrates how to apply group theory concepts to detect invariant features for brain tumor detection.

# 1. Setup and Libraries
# Ensure the necessary libraries are installed
# pip install numpy opencv-python scikit-image matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.transform import rotate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# 2. Loading and Preprocessing the Image
def load_and_preprocess_image(image_path):
    """Load and preprocess the MRI image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image


def display_image(image, title):
    """Display the image with a given title."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Load and preprocess the image
image_path = 'Y1.jpg'  # Replace with the path to your image
image = load_and_preprocess_image(image_path)
display_image(image, 'Original Brain MRI')

yes_folder = '/Users/aniruddhmodi/PycharmProjects/Research/archive/brain_tumor_dataset/yes'
no_folder = ''
# 3. Applying Group Theoretical Transformations
def extract_hog_features(image):
    """Extract Histogram of Oriented Gradients (HOG) features from the image."""
    features, hog_image = hog(image, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    return features, hog_image


def rotate_and_extract(image, angles):
    """Rotate the image by specified angles and extract invariant features."""
    invariant_features = []
    for angle in angles:
        rotated_image = rotate(image, angle, resize=False)
        features, _ = extract_hog_features(rotated_image)
        invariant_features.append(features)
    return invariant_features


# Define a group of rotations (0, 90, 180, 270 degrees)
angles = [0, 90, 180, 270]
invariant_features = rotate_and_extract(image, angles)

# Display the rotated images and their HOG features
fig, axs = plt.subplots(2, len(angles), figsize=(12, 6))
for i, angle in enumerate(angles):
    rotated_image = rotate(image, angle, resize=False)
    _, hog_image = extract_hog_features(rotated_image)

    axs[0, i].imshow(rotated_image, cmap='gray')
    axs[0, i].set_title(f'Rotated {angle}°')
    axs[0, i].axis('off')

    axs[1, i].imshow(hog_image, cmap='gray')
    axs[1, i].set_title('HOG Features')
    axs[1, i].axis('off')

plt.show()


# 4. Analysis of Invariant Features
def print_features_summary(invariant_features, angles):
    """Print the feature vectors for different rotations."""
    for i, features in enumerate(invariant_features):
        print(f"Feature vector for rotation {angles[i]}°:")
        print(features[:10])  # Print first 10 elements of the feature vector for brevity


print_features_summary(invariant_features, angles)


# 5. Modeling the Invariant Features Using Group Theory
def simulate_dataset(num_samples, feature_length):
    """Simulate a dataset for demonstration purposes."""
    X = np.random.rand(num_samples, feature_length)
    y = np.random.randint(0, 2, num_samples)
    return X, y


# Simulate a larger set of invariant features (For real use, load a proper dataset)
# Assume X contains HOG features for each image, and y contains corresponding labels
X, y = simulate_dataset(100, len(invariant_features[0]))  # 100 samples with the same feature length

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
