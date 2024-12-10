import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from data_preparation import prepare_data
from model import build_model
from train import train_model
from evaluate import evaluate_model

def test_model_with_images(model_path, images_folder):
    """
    Tests a trained model with multiple images from a folder.

    Args:
        model_path (str): Path to the trained model file.
        images_folder (str): Path to the folder containing test images.

    Returns:
        None: Prints predictions for each image.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Iterate through all images in the folder
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
            # Load and preprocess the image
            image = load_img(image_path, target_size=(128, 128))
            image_array = img_to_array(image) / 255.0  # Normalize the image
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(image_array)
            result = "Tumor" if prediction[0][0] > 0.5 else "No Tumor"
            print(f"Prediction for {image_name}: {result}")
        else:
            print(f"Skipping non-image file: {image_name}")

if __name__ == "__main__":
    DATA_DIR = "dataset/brain_tumor_dataset"
    MODEL_PATH = "brain_tumor_detection_model.h5"
    TEST_IMAGES_FOLDER = "test_images"  # Replace with the path to your test images folder

    # Step 1: Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(DATA_DIR)

    # Step 2: Build and train the model
    print("Building and training the model...")
    model = build_model()
    train_model(model, X_train, y_train)

    # Step 3: Evaluate the model on the test set
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Step 4: Save the model
    print(f"Saving the model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    # Step 5: Test the model with images in a folder
    print(f"Testing the model with images in folder: {TEST_IMAGES_FOLDER}...")
    test_model_with_images(MODEL_PATH, TEST_IMAGES_FOLDER)
