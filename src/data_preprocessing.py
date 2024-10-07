import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    # Load image
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    return img_array

def prepare_dataset(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    
    # Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8, 1.2),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Apply Gaussian and Median Filtering
    def apply_filters(image):
        image = tf.image.adjust_contrast(image, 2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image
    
    # Modify the loop to include data augmentation and filtering
    for label, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_array = load_and_preprocess_image(img_path, target_size)
            img_array = apply_filters(img_array)
            images.append(img_array)
            labels.append(label)
    
    # Adjust data splitting ratios
    X_train, X_temp, y_train, y_temp = train_test_split(np.array(images), np.array(labels), test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
if __name__ == "__main__":
    data_dir = "path/to/your/mri/images"
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(data_dir)
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")