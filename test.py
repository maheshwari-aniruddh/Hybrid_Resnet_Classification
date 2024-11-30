import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Parameters for LBP
radius = 2  # Radius of circle
n_points = 8 * radius  # Number of points for LBP


def visualize_lbp(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found.")
        return

    # Compute LBP
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")

    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Grayscale Image')

    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP Image')

    plt.show()


# Provide the path to your image file
image_path = '115.png'
visualize_lbp(image_path)
