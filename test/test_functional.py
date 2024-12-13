import unittest
import os
from model import build_model
from data_preparation import prepare_data
from evaluate import evaluate_model
from train import train_model
from main import test_model_with_images
import tensorflow as tf
from test.TestUtils import TestUtils

class FunctionalTest(unittest.TestCase):
    def setUp(self):
        """Initialize the setup for the tests."""
        # Initialize TestUtils object
        self.test_obj = TestUtils()

        # Paths
        self.model_file = "brain_tumor_detection_model.h5"
        self.test_images_dir = "test_images"
        self.data_dir = "dataset/brain_tumor_dataset"

        # Prepare data dynamically
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data(self.data_dir)

        # Build and train the model
        self.model = build_model()
        self.train_history = train_model(self.model, self.X_train, self.y_train)

    # 1. Test if the model builds successfully
    def test_model_build(self):
        """Test if the CNN model builds successfully."""
        try:
            model = build_model()
            model_built = True
        except Exception as e:
            print(f"Model build error: {e}")
            model_built = False

        self.test_obj.yakshaAssert("TestModelBuild", model_built, "functional")
        self.assertTrue(model_built, "Model did not build successfully.")

    # 2. Test if the model file is saved
    def test_model_file_saved(self):
        """Test if the model is saved with the correct filename."""
        self.model.save(self.model_file)
        model_saved = os.path.exists(self.model_file)
        self.test_obj.yakshaAssert("TestModelFileSaved", model_saved, "functional")
        self.assertTrue(model_saved, "Model file was not saved.")

    # 3. Test data preparation
    def test_data_preparation(self):
        """Test if data preparation returns non-empty data."""
        self.assertIsNotNone(self.X_train, "X_train is None.")
        self.assertIsNotNone(self.X_test, "X_test is None.")
        self.assertIsNotNone(self.y_train, "y_train is None.")
        self.assertIsNotNone(self.y_test, "y_test is None.")
        self.test_obj.yakshaAssert("TestDataPreparation", True, "functional")

    # 4. Test model accuracy after training
    def test_model_accuracy(self):
        """Test if the model achieves the expected accuracy."""
        _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        expected_accuracy = 0.80  # Adjust this as necessary
        self.test_obj.yakshaAssert("TestModelAccuracy", accuracy >= expected_accuracy, "functional")
        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy {accuracy} is less than expected {expected_accuracy}.")

    # 5. Test if the model runs for 10 epochs
    def test_model_epochs(self):
        """Test if the model runs for the expected number of epochs."""
        actual_epochs = len(self.train_history.epoch)
        expected_epochs = 10
        self.test_obj.yakshaAssert("TestModelEpochs", actual_epochs == expected_epochs, "functional")
        self.assertEqual(actual_epochs, expected_epochs, f"Model ran for {actual_epochs} epochs, expected {expected_epochs}.")

    # 6. Test the confusion matrix
    def test_confusion_matrix(self):
        """Test if the confusion matrix matches the expected values."""
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        confusion_matrix = tf.math.confusion_matrix(self.y_test, y_pred).numpy().tolist()
        expected_matrix = [[14, 3], [2, 16]]  # Adjust dynamically if needed
        self.test_obj.yakshaAssert("TestConfusionMatrix", confusion_matrix == expected_matrix, "functional")
        self.assertEqual(confusion_matrix, expected_matrix, f"Confusion Matrix {confusion_matrix} does not match expected {expected_matrix}.")

    # 7. Test predictions for test images
    def test_image_predictions(self):
        """Test predictions for images in the test_images folder."""
        predictions = {}
        for image in os.listdir(self.test_images_dir):
            image_path = os.path.join(self.test_images_dir, image)
            prediction = test_model_with_images(self.model_file, self.test_images_dir)
            predictions[image] = prediction

        expected_predictions = {
            "3 no.jpg": "No Tumor",
            "4 no.jpg": "No Tumor",
            "Y7.jpg": "No Tumor",
            "Y8.jpg": "Tumor",
            "Y9.jpg": "Tumor"
        }
        predictions_matched = predictions == expected_predictions
        self.test_obj.yakshaAssert("TestImagePredictions", predictions_matched, "functional")
        self.assertEqual(predictions, expected_predictions, f"Predictions {predictions} do not match expected {expected_predictions}.")

if __name__ == "__main__":
    unittest.main()
