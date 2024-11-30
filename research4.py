import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from scipy.fftpack import fft
from scipy.interpolate import interp1d

IMAGE_SIZE = (128, 128)
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 50
NUM_FOLDS = 5
NUM_HARMONICS = 10  # Number of harmonics for EFD


# Load and preprocess images
def load_images_from_folder(folder_path, size=IMAGE_SIZE):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
            filenames.append(filename)
        else:
            print(f"Failed to load image: {filename}")
    return np.array(images), filenames


def preprocess_images(image_array):
    image_array = image_array.astype('float32') / 255.0
    return np.stack((image_array,) * 3, axis=-1)


# Extract HOG features
def extract_hog_features(image):
    features = feature.hog(image, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), visualize=False)
    return features


def extract_all_hog_features(images):
    return np.array([extract_hog_features(img) for img in images])


# EFD: Normalize and parameterize contours
def normalize_contour(contour):
    contour = np.array(contour).reshape(-1, 2)
    centroid = np.mean(contour, axis=0)
    contour -= centroid
    max_distance = np.max(np.sqrt(np.sum(contour ** 2, axis=1)))
    contour /= max_distance
    return contour


def parameterize_contour(contour, num_points=100):
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    arc_length = np.concatenate(([0], np.cumsum(distances)))
    interp_func = interp1d(arc_length, contour, kind='linear', fill_value='extrapolate')
    parameterized_contour = interp_func(np.linspace(0, arc_length[-1], num_points))
    return parameterized_contour


# Extract EFD features
def extract_efd_features(contour):
    parameterized_contour = parameterize_contour(normalize_contour(contour))
    x = parameterized_contour[:, 0]
    y = parameterized_contour[:, 1]
    fx = fft(x)
    fy = fft(y)
    efd = np.concatenate([fx[:NUM_HARMONICS], fy[:NUM_HARMONICS]])
    return np.concatenate([np.real(efd), np.imag(efd)])  # Combine real and imaginary parts


def extract_all_efd_features(images):
    efd_features_list = []
    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            efd_features = extract_efd_features(largest_contour[:, 0, :])
            efd_features_list.append(efd_features)
        else:
            efd_features_list.append(np.zeros(2 * NUM_HARMONICS * 2))  # Placeholder for no contour
    return np.array(efd_features_list)


# Build CNN model
def build_cnn_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    return Model(inputs=base_model.input, outputs=x)


# Combine CNN with EFD and HOG features
def build_combined_model(image_input_shape, hog_input_shape, efd_input_shape):
    image_input = Input(shape=image_input_shape)
    hog_input = Input(shape=(hog_input_shape,))
    efd_input = Input(shape=(efd_input_shape,))

    cnn_model = build_cnn_model(image_input_shape)(image_input)

    combined_features = Concatenate()([cnn_model, hog_input, efd_input])
    combined_features = Dense(128)(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = LeakyReLU(alpha=0.1)(combined_features)
    combined_features = Dropout(0.5)(combined_features)
    output = Dense(1, activation='sigmoid')(combined_features)

    model = Model(inputs=[image_input, hog_input, efd_input], outputs=output)
    return model


def create_data_generator(augmentation=True):
    if augmentation:
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator()


# Visualization functions
def visualize_image_and_contour(image, contour):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.plot(contour[:, 0], contour[:, 1], 'r-', lw=2)
    plt.title('Contour')

    plt.show()


def visualize_efd_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        efd_features = extract_efd_features(largest_contour[:, 0, :])

        plt.figure(figsize=(10, 5))

        # Plot EFD features
        plt.subplot(1, 2, 1)
        plt.plot(efd_features[:NUM_HARMONICS], label='Real Part')
        plt.plot(efd_features[NUM_HARMONICS:], label='Imaginary Part')
        plt.title('EFD Features')
        plt.xlabel('Harmonics')
        plt.ylabel('Magnitude')
        plt.legend()

        # Show the original image with contours
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='gray')
        plt.plot(largest_contour[:, 0, 0], largest_contour[:, 0, 1], 'r-', lw=2)
        plt.title('Contour on Image')

        plt.show()


# Load and prepare data
yes_folder = '/path/to/yes_folder'
no_folder = '/path/to/no_folder'

yes_images, _ = load_images_from_folder(yes_folder)
no_images, _ = load_images_from_folder(no_folder)

X = np.concatenate([yes_images, no_images], axis=0)
y = np.concatenate([np.ones(len(yes_images)), np.zeros(len(no_images))])

X = preprocess_images(X)
hog_features = extract_all_hog_features(X)
efd_features = extract_all_efd_features(X)

# Adjust input shape based on extracted HOG and EFD features
hog_input_shape = hog_features.shape[1]
efd_input_shape = efd_features.shape[1]

# Implement k-fold cross-validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    print(f"Training on fold {fold}")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    hog_train, hog_val = hog_features[train_index], hog_features[val_index]
    efd_train, efd_val = efd_features[train_index], efd_features[val_index]

    # Data augmentation
    train_datagen = create_data_generator(augmentation=True)
    val_datagen = create_data_generator(augmentation=False)

    # Build and compile the model
    model = build_combined_model((128, 128, 3), hog_input_shape, efd_input_shape)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        x=[X_train, hog_train, efd_train],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val, hog_val, efd_val], y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate([X_val, hog_val, efd_val], y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Predict
    predictions = model.predict([X_val, hog_val, efd_val])
    predictions_binary = (predictions > 0.5).astype(int)

    # Calculate additional metrics
    precision = precision_score(y_val, predictions_binary)
    recall = recall_score(y_val, predictions_binary)
    f1 = f1_score(y_val, predictions_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    fold_results.append({
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Print average results across all folds
avg_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0].keys()}
print("\nAverage results across all folds:")
for metric, value in avg_results.items():
    print(f"{metric}: {value}")

# Save the final model
model.save('brain_tumor_segmentation_model_fournier.h5')
print("Model saved successfully.")


# Example of using the model for prediction
def predict_tumor(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMAGE_SIZE)
    image = preprocess_images(np.array([image]))
    hog_feature = extract_hog_features(image[0]).reshape(1, -1)
    efd_feature = extract_efd_features(
        cv2.findContours(image[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0][:, 0, :]).reshape(1, -1)
    return model.predict([image, hog_feature, efd_feature])[0][0]


test_image_path = '/path/to/new/image.jpg'
tumor_probability = predict_tumor(test_image_path, model)
print(f"Probability of tumor: {tumor_probability * 100:.2f}%")
