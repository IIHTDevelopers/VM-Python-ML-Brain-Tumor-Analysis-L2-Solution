import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path, img_size=(64, 64)):
    """Preprocesses a single image for prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, img_size)
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=(0, -1))  # Reshape for CNN input

def predict_images(model_path, test_folder):
    """Predicts if images in the test_images folder have a tumor or not."""
    model = load_model(model_path)
    print("Model loaded successfully!")

    for file in os.listdir(test_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(test_folder, file)
            preprocessed_img = preprocess_image(image_path)
            prediction = model.predict(preprocessed_img)
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
            print(f"Image: {file} → Prediction: {result}")
