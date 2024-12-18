import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from dataloader import load_data, split_data, data_summary, get_image_count
from model_trainer import create_cnn_model, train_model, evaluate_model
import os
import cv2


def get_image_count(data_path):
    """Counts the total number of images in the dataset."""
    count = sum([len(files) for r, d, files in os.walk(data_path)])
    print(f"Total images in dataset: {count}")
    return count

def preprocess_image(image_path, img_size=(64, 64)):
    """Preprocesses a single image for prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, img_size)
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=(0, -1))  # Reshape for CNN input

def predict_test_folder(model_path, test_folder):
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

if __name__ == "__main__":
    # Define paths
    DATA_PATH = "brain_tumor_dataset"
    MODEL_PATH = "brain_tumor_cnn_model.h5"
    TEST_FOLDER = "test_images"

    # Step 1: Load and preprocess the data
    print("Loading data...")
    X, y = load_data(DATA_PATH)
    X = X / 255.0  # Normalize images
    X = X.reshape(-1, 64, 64, 1)  # Add channel dimension
    y = np.array(y)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    data_summary(X_train, X_test, y_train, y_test)

    # Step 3: Create and train model
    print("Creating model...")
    model = create_cnn_model()
    print("Training model...")
    train_model(model, X_train, y_train)

    # Step 4: Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Step 5: Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Step 7: Count total images
    get_image_count(DATA_PATH)

    # Step 8: Predict images in test_images folder
    print("\nChecking test_images folder images...")
    predict_test_folder(MODEL_PATH, TEST_FOLDER)
