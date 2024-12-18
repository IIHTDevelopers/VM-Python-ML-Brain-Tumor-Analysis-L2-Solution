import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """Loads images and labels from the dataset path."""
    images = []
    labels = []
    for label, condition in enumerate(["no", "yes"]):
        folder_path = os.path.join(data_path, condition)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64))
            images.append(img_resized)
            labels.append(label)
    return np.array(images), np.array(labels)

def split_data(X, y, test_size=0.2):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def data_summary(X_train, X_test, y_train, y_test):
    """Prints summary of data."""
    print(f"Total training samples: {len(X_train)}")
    print(f"Total testing samples: {len(X_test)}")
    print(f"Shape of input data: {X_train[0].shape}")

def get_image_count(data_path):
    """Counts the total number of images in the dataset."""
    count = 0
    for condition in ["no", "yes"]:
        folder_path = os.path.join(data_path, condition)
        count += len(os.listdir(folder_path))
    print(f"Total images in dataset: {count}")
    return count
