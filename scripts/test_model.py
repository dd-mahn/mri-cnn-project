import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Thêm thư mục gốc của dự án vào đường dẫn Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_preprocessing import load_and_preprocess_image, apply_filters, extract_image_attributes
from src.model import build_resnet

def load_model_and_labels():
    model_dir = os.path.join(project_root, 'model')
    model = tf.keras.models.load_model(os.path.join(model_dir, 'mri_classification_model.h5'))
    label_to_index = np.load(os.path.join(model_dir, 'label_to_index.npy'), allow_pickle=True).item()
    index_to_label = {v: k for k, v in label_to_index.items()}
    return model, index_to_label

def predict_image(model, image_path, index_to_label):
    img = load_and_preprocess_image(image_path)
    img = apply_filters(img)
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = index_to_label[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

def test_model(image_path):
    model, index_to_label = load_model_and_labels()
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    img_filtered = apply_filters(img)
    
    # Extract image attributes
    attributes = extract_image_attributes(img)
    
    # Make prediction
    predicted_label, confidence = predict_image(model, image_path, index_to_label)
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_filtered, cmap='gray')
    plt.title(f'Predicted: {predicted_label}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Image Attributes:")
    for key, value in attributes.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    image_path = input("Enter the path to the MRI image: ")
    test_model(image_path)
