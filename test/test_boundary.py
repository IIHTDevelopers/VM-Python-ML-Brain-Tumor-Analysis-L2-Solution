import unittest
from test.TestUtils import TestUtils
from main import evaluate_model, load_data, split_data, create_cnn_model, train_model

class BoundaryTest(unittest.TestCase):

    def setUp(self):
        """Setup for boundary tests."""
        self.test_obj = TestUtils()
        self.minimum_accuracy = 0.8  # Accuracy threshold
        self.maximum_loss = 0.5      # Loss threshold
        self.data_path = "brain_tumor_dataset"
        self.epochs_to_check = 5  # Expected epochs for training
        self.batch_size = 16

        # Prepare dataset for training and testing
        X, y = load_data(self.data_path)
        self.X = X / 255.0  # Normalize
        self.X = self.X.reshape(-1, 64, 64, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.X, y)

    def test_accuracy_boundary(self):
        """Test if the model accuracy is not less than 0.8."""
        # Create and train the model
        model = create_cnn_model()
        train_model(model, self.X_train, self.y_train, epochs=self.epochs_to_check, batch_size=self.batch_size)

        # Evaluate the model and get accuracy
        _, accuracy = evaluate_model(model, self.X_test, self.y_test)
        is_valid = accuracy >= self.minimum_accuracy

        self.test_obj.yakshaAssert("TestAccuracyBoundary", is_valid, "boundary")
        print(f"Accuracy Boundary Test: {accuracy:.4f} (Threshold: {self.minimum_accuracy}) → {'Passed' if is_valid else 'Failed'}")

    def test_loss_boundary(self):
        """Test if the model loss is not greater than 0.5."""
        # Create and train the model
        model = create_cnn_model()
        train_model(model, self.X_train, self.y_train, epochs=self.epochs_to_check, batch_size=self.batch_size)

        # Evaluate the model and get loss
        loss, _ = evaluate_model(model, self.X_test, self.y_test)
        is_valid = loss <= self.maximum_loss

        self.test_obj.yakshaAssert("TestLossBoundary", is_valid, "boundary")
        print(f"Loss Boundary Test: {loss:.4f} (Threshold: {self.maximum_loss}) → {'Passed' if is_valid else 'Failed'}")

if __name__ == "__main__":
    unittest.main()
