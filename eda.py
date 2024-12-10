import numpy as np
import pandas as pd

def analyze_data(X, y):
    class_distribution = pd.Series(y).value_counts()
    avg_pixel_intensity = {label: np.mean(X[y == label]) for label in np.unique(y)}
    total_samples = len(y)
    class_ratios = {label: count / total_samples for label, count in class_distribution.items()}
    unique_images = np.unique(X.reshape(X.shape[0], -1), axis=0)
    duplicates = len(X) - len(unique_images)
    print("EDA Results:")
    print("1. Class Distribution:\n", class_distribution)
    print("2. Average Pixel Intensity per Class:\n", avg_pixel_intensity)
    print("3. Class Ratios:\n", class_ratios)
    print(f"4. Number of Duplicate Images: {duplicates}")
    print(f"5. Image Dimensions: {X.shape[1:]}")
    print(f"6. Training Samples: {int(total_samples * 0.8)}, Testing Samples: {int(total_samples * 0.2)}")
    return class_distribution, avg_pixel_intensity, class_ratios
