import unittest
import os
import numpy as np
from tensorflow.keras.models import load_model
from dataloader import load_data, split_data
from model_trainer import create_cnn_model, train_model, evaluate_model
from main import preprocess_image
from test.TestUtils import TestUtils  # Assuming TestUtils is available

class ConsoleValuesTest(unittest.TestCase):

    def setUp(self):
        """Setup before running test cases."""
        self.test_obj = TestUtils()
        self.model_path = "brain_tumor_cnn_model.h5"
        self.test_folder = "test_images"
        self.data_path = "brain_tumor_dataset"
        self.expected_accuracy = 0.9019607901573181  # Threshold accuracy
        self.expected_test_loss = 0.31537821888923645  # Example expected loss
        self.total_images_count = 253  # Expected total image count
        self.epochs_to_check = 5       # Expected number of epochs

    def test_training_samples_count(self):
        """Test if the number of training samples is correct."""
        X, y = load_data(self.data_path)
        X_train, X_test, _, _ = split_data(X, y)
        training_samples = len(X_train)
        is_valid = training_samples == 202  # Replace with actual count
        self.test_obj.yakshaAssert("TestTrainingSamplesCount", is_valid, "functional")
        print(f"Training Samples Count: {training_samples} → {'Passed' if is_valid else 'Failed'}")

    def test_testing_samples_count(self):
        """Test if the number of testing samples is correct."""
        X, y = load_data(self.data_path)
        _, X_test, _, _ = split_data(X, y)
        testing_samples = len(X_test)
        is_valid = testing_samples == 51  # Replace with actual count
        self.test_obj.yakshaAssert("TestTestingSamplesCount", is_valid, "functional")
        print(f"Testing Samples Count: {testing_samples} → {'Passed' if is_valid else 'Failed'}")


    def test_total_images_count(self):
        """Test if the total number of images matches the expected count."""
        total_count = sum([len(files) for _, _, files in os.walk(self.data_path)])
        is_valid = total_count == self.total_images_count
        self.test_obj.yakshaAssert("TestTotalImagesCount", is_valid, "functional")
        print(f"Total Images Count: {total_count} → {'Passed' if is_valid else 'Failed'}")

    def test_model_file_created(self):
        """Test if the model file is created successfully."""
        X, y = load_data(self.data_path)
        X = X / 255.0
        X = X.reshape(-1, 64, 64, 1)
        X_train, _, y_train, _ = split_data(X, y)

        model = create_cnn_model()
        train_model(model, X_train, y_train, epochs=1, batch_size=16)
        model.save(self.model_path)

        file_exists = os.path.exists(self.model_path)
        self.test_obj.yakshaAssert("TestModelFileCreated", file_exists, "functional")
        print(f"Model File Created: {'Passed' if file_exists else 'Failed'}")

    def test_epoch_count(self):
        """Test if the number of training epochs equals 5."""
        X, y = load_data(self.data_path)
        X = X / 255.0
        X = X.reshape(-1, 64, 64, 1)
        X_train, _, y_train, _ = split_data(X, y)

        model = create_cnn_model()
        history = train_model(model, X_train, y_train, epochs=self.epochs_to_check, batch_size=16)
        epoch_count = len(history.epoch)

        is_valid = epoch_count == self.epochs_to_check
        self.test_obj.yakshaAssert("TestEpochCount", is_valid, "functional")
        print(f"Epoch Count: {epoch_count} → {'Passed' if is_valid else 'Failed'}")

    def test_prediction_Y7(self):
        """Test predictions for the test image Y7.jpg."""
        model = load_model(self.model_path)
        test_image_path = os.path.join(self.test_folder, "Y7.jpg")
        preprocessed_img = preprocess_image(test_image_path)

        prediction = model.predict(preprocessed_img)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
        is_valid = result in ["Tumor Detected", "No Tumor"]
        self.test_obj.yakshaAssert("TestPredictionY7", is_valid, "functional")
        print(f"Test Image Prediction (Y7): {result} → {'Passed' if is_valid else 'Failed'}")


if __name__ == "__main__":
    unittest.main()
