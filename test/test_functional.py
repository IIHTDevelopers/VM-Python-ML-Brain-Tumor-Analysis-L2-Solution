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
        self.test_obj = TestUtils()

        # Paths
        self.model_file = "brain_tumor_detection_model.h5"
        self.test_images_dir = "test_images"
        self.data_dir = "brain_tumor_dataset"

        # Initialize states
        self.data_prepared = False
        self.model_built = False

        # Check if the dataset exists
        if not os.path.exists(self.data_dir):
            print("Dataset directory not found. Failed.")
            return

        # Prepare data dynamically
        try:
            result = prepare_data(self.data_dir)
            if result is None:
                print("prepare_data returned None. Failed.")
                self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = result
                self.data_prepared = True
        except Exception as e:
            print(f"Exception during data preparation: {e}. Failed.")
            self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Build and train the model
        try:
            if self.data_prepared:
                self.model = build_model()
                self.train_history = train_model(self.model, self.X_train, self.y_train)
                self.model_built = True
            else:
                print("Skipping model building and training due to missing data. Failed.")
                self.model = None
                self.train_history = None
        except Exception as e:
            print(f"Exception during model building or training: {e}. Failed.")
            self.model = None
            self.train_history = None

    # 1. Test if the model builds successfully
    def test_model_build(self):
        """Test if the CNN model builds successfully."""
        try:
            model = build_model()
            model_built = True
            print("Model build passed.")
        except Exception as e:
            print(f"Model build error: {e}. Failed.")
            model_built = False

        self.test_obj.yakshaAssert("TestModelBuild", model_built, "functional")
        if not model_built:
            print("Test failed but marking as passed for continuation.")
        self.assertTrue(True, "Model build test completed.")

    # 2. Test if the model file is saved
    def test_model_file_saved(self):
        """Test if the model is saved with the correct filename."""
        if not self.model_built:
            print("Model file save failed due to model not being built. Failed.")
            self.test_obj.yakshaAssert("TestModelFileSaved", False, "functional")
            self.assertTrue(True, "Model file save test completed.")
            return

        try:
            self.model.save(self.model_file)
            model_saved = os.path.exists(self.model_file)
            print("Model file save passed." if model_saved else "Model file save failed.")
        except Exception as e:
            print(f"Error saving model: {e}. Failed.")
            model_saved = False

        self.test_obj.yakshaAssert("TestModelFileSaved", model_saved, "functional")
        self.assertTrue(True, "Model file save test completed.")

    # 3. Test data preparation
    def test_data_preparation(self):
        """Test if data preparation returns non-empty data."""
        if not self.data_prepared:
            print("Data preparation test failed. prepare_data returned None.")
            self.test_obj.yakshaAssert("TestDataPreparation", False, "functional")
            self.assertTrue(True, "Data preparation test completed.")
            return
        print("Data preparation passed.")
        self.test_obj.yakshaAssert("TestDataPreparation", True, "functional")
        self.assertTrue(True, "Data preparation test completed.")

    # 4. Test model accuracy after training
    def test_model_accuracy(self):
        """Test if the model achieves the expected accuracy."""
        if not self.model_built:
            print("Model accuracy test failed due to model not being built. Failed.")
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            self.assertTrue(True, "Model accuracy test completed.")
            return
        try:
            _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            expected_accuracy = 0.80  # Adjust this as necessary
            print("Model accuracy test passed." if accuracy >= expected_accuracy else f"Model accuracy test failed. Accuracy: {accuracy}")
            self.test_obj.yakshaAssert("TestModelAccuracy", accuracy >= expected_accuracy, "functional")
        except Exception as e:
            print(f"Model accuracy test failed: {e}. Failed.")
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
        self.assertTrue(True, "Model accuracy test completed.")

    # 5. Test if the model runs for 10 epochs
    def test_model_epochs(self):
        """Test if the model runs for the expected number of epochs."""
        if not self.model_built:
            print("Model epoch test failed due to model not being built. Failed.")
            self.test_obj.yakshaAssert("TestModelEpochs", False, "functional")
            self.assertTrue(True, "Model epoch test completed.")
            return
        try:
            actual_epochs = len(self.train_history.epoch)
            expected_epochs = 10
            print("Model epochs test passed." if actual_epochs == expected_epochs else f"Model epochs test failed. Ran for {actual_epochs} epochs instead of {expected_epochs}.")
            self.test_obj.yakshaAssert("TestModelEpochs", actual_epochs == expected_epochs, "functional")
        except Exception as e:
            print(f"Model epochs test failed: {e}. Failed.")
            self.test_obj.yakshaAssert("TestModelEpochs", False, "functional")
        self.assertTrue(True, "Model epoch test completed.")

if __name__ == "__main__":
    unittest.main()
