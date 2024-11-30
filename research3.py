import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Parameters
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 50
NUM_FOLDS = 5

# Custom callback to log the learning rate
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)  # Access learning rate
        self.model.history.history.setdefault('lr', []).append(lr)  # Store in history

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

# Function to load images
def load_images_from_folder(folder_path, size=IMAGE_SIZE):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cropped_img = crop_image_based_on_threshold(img)
            img_resized = cv2.resize(cropped_img, size)
            img_resized_3d = np.stack((img_resized,) * 3, axis=-1)
            images.append(img_resized_3d)
            filenames.append(filename)
        else:
            print(f"Failed to load image: {filename}")
    return np.array(images), filenames

# Preprocess images (normalizing)
def preprocess_images(image_array):
    return image_array.astype('float32') / 255.0

# Extract HOG features
def extract_hog_features(image):
    gray_image = image[:, :, 0]
    features = feature.hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# Extract LBP features
def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = image[:, :, 0]
    lbp = feature.local_binary_pattern(gray_image, num_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Build the CNN model using ResNet50 as the base
def build_cnn_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    return Model(inputs=base_model.input, outputs=x)

# Build the combined CNN, HOG, and LBP model
def build_combined_model(image_input_shape, hog_input_shape, lbp_input_shape):
    image_input = Input(shape=image_input_shape)
    hog_input = Input(shape=(hog_input_shape,))
    lbp_input = Input(shape=(lbp_input_shape,))

    cnn_model = build_cnn_model(image_input_shape)(image_input)

    combined_features = Concatenate()([cnn_model, hog_input, lbp_input])
    combined_features = Dense(128)(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = LeakyReLU(alpha=0.1)(combined_features)
    combined_features = Dropout(0.5)(combined_features)
    output = Dense(1, activation='sigmoid')(combined_features)

    model = Model(inputs=[image_input, hog_input, lbp_input], outputs=output)
    return model

# Data generator with augmentation
def create_data_generator(augmentation=True):
    if augmentation:
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator()

# Paths to your data
yes_folder = '/Users/aniruddhmodi/PycharmProjects/Research/brd/yes'
no_folder = '/Users/aniruddhmodi/PycharmProjects/Research/brd/no'

# Load and preprocess data
yes_images, yes_filenames = load_images_from_folder(yes_folder)
no_images, no_filenames = load_images_from_folder(no_folder)

X = np.concatenate([yes_images, no_images], axis=0)
y = np.concatenate([np.ones(len(yes_images)), np.zeros(len(no_images))])

X = preprocess_images(X)
hog_features = np.array([extract_hog_features(img) for img in X])
lbp_features = np.array([extract_lbp_features(img) for img in X])

# Adjust input shape based on extracted HOG and LBP features
hog_input_shape = hog_features.shape[1]
lbp_input_shape = lbp_features.shape[1]
# Implement k-fold cross-validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# To store metrics across folds
all_train_accuracy = []
all_val_accuracy = []
all_train_loss = []  # New list to store training loss
all_val_loss = []
all_learning_rates = []
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    print(f"Training on fold {fold}")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]  # Fixed line

    hog_train, hog_val = hog_features[train_index], hog_features[val_index]
    lbp_train, lbp_val = lbp_features[train_index], lbp_features[val_index]

    # Build and compile the model
    model = build_combined_model((128, 128, 3), hog_input_shape, lbp_input_shape)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
    lr_logger = LearningRateLogger()

    # Start timer
    start_time = time.time()

    # Train the model
    history = model.fit(
        x=[X_train, hog_train, lbp_train],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val, hog_val, lbp_val], y_val),
        callbacks=[early_stopping, reduce_lr, lr_logger]
    )

    # End timer
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training duration for fold {fold}: {training_duration:.2f} seconds")

    # Store metrics for averaging
    all_train_accuracy.append(history.history['accuracy'])
    all_val_accuracy.append(history.history['val_accuracy'])
    all_val_loss.append(history.history['val_loss'])
    all_train_loss.append(history.history['loss'])

    all_learning_rates.append(history.history['lr'])  # Log learning rates

# Print metrics for each fold
for fold in range(NUM_FOLDS):
    print(f"Metrics for Fold {fold + 1}:")
    print(f"Train Accuracy: {all_train_accuracy[fold][-1]:.4f}")
    print(f"Validation Accuracy: {all_val_accuracy[fold][-1]:.4f}")
    print(f"Train Loss: {all_train_loss[fold][-1]:.4f}")
    print(f"Validation Loss: {all_val_loss[fold][-1]:.4f}")
    print(f"Final Learning Rate: {all_learning_rates[fold][-1]:.6f}")
    print("-" * 50)

# Print averaged metrics across all folds (optional)
print("Averaged Metrics across all folds:")
print(f"Average Train Accuracy: {np.mean([train[-1] for train in all_train_accuracy]):.4f}")
print(f"Average Validation Accuracy: {np.mean([val[-1] for val in all_val_accuracy]):.4f}")
print(f"Average Train Loss: {np.mean([loss[-1] for loss in all_train_loss]):.4f}")
print(f"Average Validation Loss: {np.mean([val[-1] for val in all_val_loss]):.4f}")
print(f"Average Final Learning Rate: {np.mean([lr[-1] for lr in all_learning_rates]):.6f}")

# Average metrics across all folds for each epoch
min_epochs = min(len(train) for train in all_train_accuracy)
avg_train_accuracy = np.mean([train[:min_epochs] for train in all_train_accuracy], axis=0)
avg_val_accuracy = np.mean([val[:min_epochs] for val in all_val_accuracy], axis=0)
avg_train_loss = np.mean([loss[:min_epochs] for loss in all_train_loss], axis=0)  # Average training loss
avg_val_loss = np.mean([val[:min_epochs] for val in all_val_loss], axis=0)
avg_learning_rates = np.mean([lr[:min_epochs] for lr in all_learning_rates], axis=0)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(avg_train_accuracy, label='Train Accuracy')
plt.plot(avg_val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot validation loss values
plt.figure(figsize=(12, 6))
plt.plot(avg_val_loss, label='Validation Loss')
plt.title('Model Validation Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot learning rate over epochs
plt.figure(figsize=(12, 6))
plt.plot(avg_learning_rates, label='Learning Rate')
plt.title('Learning Rate per Epoch')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(avg_train_loss, label='Train Loss')
plt.plot(avg_val_loss, label='Validation Loss')
plt.title('Training Loss vs Validation Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
