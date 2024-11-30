import numpy as np
import cv2
from skimage import feature
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = 'brain_tumor_detection_model.h5'
IMAGE_PATH = '11 no.jpg'

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    return image

# Function to extract HOG features


# Function to extract HOG features
from skimage.feature import hog
import numpy as np
import cv2


# Function to extract HOG features
def extract_hog_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust these parameters to match the feature size expected by the model
    hog_features = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True  # Ensure feature_vector is True
    )

    # Check if the length matches expected length
    expected_length = 1764
    if len(hog_features) != expected_length:
        print(f"Warning: HOG feature length ({len(hog_features)}) does not match expected length ({expected_length})")
        # Optionally resize or process features to match the expected length
    return hog_features


# Function to extract EFD features
def extract_efd_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    efd_features = np.mean(magnitude_spectrum, axis=0)  # Modify if needed
    # Assuming the model expects EFD features to be of certain length
    return efd_features

# Load the model
model = load_model(MODEL_PATH)

# Print model summary to verify input shapes
model.summary()

# Preprocess the new image
image = preprocess_image(IMAGE_PATH)
hog_features = extract_hog_features(image)
efd_features = extract_efd_features(image)

# Prepare the input for the model
image_input = np.expand_dims(image, axis=0)
hog_input = np.expand_dims(hog_features, axis=0)
efd_input = np.expand_dims(efd_features, axis=0)

# Check shapes before predicting
print(f"Image input shape: {image_input.shape}")
print(f"HOG input shape: {hog_input.shape}")
print(f"EFD input shape: {efd_input.shape}")

# Make prediction
prediction = model.predict([image_input, hog_input, efd_input])
probability = prediction[0][0]  # Get the probability value

# Convert probability to percentage
percentage_chance = probability * 100

# Print result
print(f"Percentage chance of tumor: {percentage_chance:.2f}%")
