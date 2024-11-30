import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure

# Load image
image_path = '115.png'  # Replace with your image path
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Compute HOG features and visualize
hog_features, hog_image = hog(image,
                              orientations=9,
                              pixels_per_cell=(2,2),
                              cells_per_block=(1, 1),
                              block_norm='L2-Hys',
                              visualize=True)

# Rescale the HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Create a side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Display the original image
ax1.axis('off')
ax1.imshow(image1, cmap='gray')
ax1.set_title('Original Image')

# Display the HOG visualization
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap='gray')
ax2.set_title('HOG Visualization')

# Save the comparison plot to a file
output_path = 'hog_comparison.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Comparison saved to {output_path}")
