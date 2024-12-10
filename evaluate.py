from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int)
    print("Test Accuracy:", test_accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred
