import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy
import scipy.signal as sig
def load_and_preprocess_filt_image(filt_image_path):
    """Load and preprocess the MRI filt_image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (500,500))
    image = image / 255.0

    return image

def display_image(image, title):
    """Display the image with a given title."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
image_path = 'MRI_of_the_brain_high_es.jpg'  # Replace with the path to your image
image = load_and_preprocess_image(image_path)
display_image(image, 'Original Brain MRI')

k_x=np.array([[-10,0,10],[-10,0,10],[-2,0,2]])
k_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
G_x=sig.convolve2d(image,k_x,mode='same')
G_y=sig.convolve2d(image,k_y,mode='same')

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("Gx")
ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel("Gy")
plt.show()

