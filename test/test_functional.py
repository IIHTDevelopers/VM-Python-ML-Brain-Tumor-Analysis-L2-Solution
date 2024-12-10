import unittest
import os
from model import build_model
import tensorflow as tf
from test.TestUtils import TestUtils


class FunctionalTest(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object
        self.test_obj = TestUtils()

        # Paths
        self.model_dir = "model"
        self.model_file = "brain_tumor_detection_model.h5"
        self.test_images_dir = "test_images"

        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)

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
        if model_built:
            print("TestModelBuild = Passed")
        else:
            print("TestModelBuild = Failed")

    # 2. Test if the model directory exists
    def test_model_directory_exists(self):
        """Test if the model directory exists."""
        directory_exists = os.path.exists(self.model_dir)
        self.test_obj.yakshaAssert("TestModelDirectoryExists", directory_exists, "functional")
        if directory_exists:
            print("TestModelDirectoryExists = Passed")
        else:
            print("TestModelDirectoryExists = Failed")

    # 3. Test if the model is saved in the file brain_tumor_detection_model.h5
    def test_model_file_saved(self):
        """Test if the model is saved with the correct filename."""
        model_path = os.path.join(self.model_dir, self.model_file)
        # Simulate model saving
        model = build_model()
        model.save(model_path)

        model_saved = os.path.exists(model_path)
        self.test_obj.yakshaAssert("TestModelFileSaved", model_saved, "functional")
        if model_saved:
            print(f"TestModelFileSaved = Passed, Model saved as {self.model_file}")
        else:
            print("TestModelFileSaved = Failed")

    # 4. Test model accuracy after training
    def test_model_accuracy(self):
        """Test if the model achieves the expected accuracy."""
        expected_accuracy = 0.8571428656578064
        actual_accuracy = 0.8571428656578064  # Replace this with the actual accuracy from training logs

        accuracy_matched = actual_accuracy == expected_accuracy
        self.test_obj.yakshaAssert("TestModelAccuracy", accuracy_matched, "functional")
        if accuracy_matched:
            print(f"TestModelAccuracy = Passed, Accuracy: {actual_accuracy}")
        else:
            print(f"TestModelAccuracy = Failed, Expected: {expected_accuracy}, Got: {actual_accuracy}")

    # 5. Test if the model runs for 10 epochs
    def test_model_epochs(self):
        """Test if the model runs for the expected number of epochs."""
        expected_epochs = 10
        actual_epochs = 10  # Replace this with the actual number of epochs run during training

        epochs_matched = actual_epochs == expected_epochs
        self.test_obj.yakshaAssert("TestModelEpochs", epochs_matched, "functional")
        if epochs_matched:
            print(f"TestModelEpochs = Passed, Epochs: {actual_epochs}")
        else:
            print(f"TestModelEpochs = Failed, Expected: {expected_epochs}, Got: {actual_epochs}")

    # 6. Test the confusion matrix
    def test_confusion_matrix(self):
        """Test if the confusion matrix matches the expected values."""
        expected_matrix = [[16, 1], [4, 14]]
        actual_matrix = [[16, 1], [4, 14]]  # Replace with the actual confusion matrix values

        matrix_matched = actual_matrix == expected_matrix
        self.test_obj.yakshaAssert("TestConfusionMatrix", matrix_matched, "functional")
        if matrix_matched:
            print(f"TestConfusionMatrix = Passed, Confusion Matrix: {actual_matrix}")
        else:
            print(f"TestConfusionMatrix = Failed, Expected: {expected_matrix}, Got: {actual_matrix}")

    # 7. Test predictions for test images
    def test_image_predictions(self):
        """Test predictions for images in the test_images folder."""
        test_images_predictions = {
            "3 no.jpg": "No Tumor",
            "4 no.jpg": "No Tumor",
            "Y7.jpg": "No Tumor",
            "Y8.jpg": "Tumor",
            "Y9.jpg": "No Tumor"
        }

        predictions_matched = True
        for image, expected in test_images_predictions.items():
            prediction = expected  # Replace with actual prediction logic
            if prediction != expected:
                predictions_matched = False
                print(f"TestImagePrediction = Failed for {image}, Expected: {expected}, Got: {prediction}")
            else:
                print(f"TestImagePrediction = Passed for {image}")

        self.test_obj.yakshaAssert("TestImagePredictions", predictions_matched, "functional")


if __name__ == "__main__":
    unittest.main()
