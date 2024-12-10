import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images, labels = [], []
    for label in ['yes', 'no']:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith('.jpg'):
                image_path = os.path.join(label_dir, file)
                image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
                image = tf.keras.utils.img_to_array(image) / 255.0
                images.append(image)
                labels.append(1 if label == 'yes' else 0)
    return np.array(images), np.array(labels)

def prepare_data(data_dir):
    X, y = load_data(data_dir)
    return train_test_split(X, y, test_size=0.2, random_state=42)
