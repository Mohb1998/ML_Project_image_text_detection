import os
import random
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def fetch_data(directory_path, limit=10000, image_size=(32, 32)):
    """Fetch images and corresponding labels from the directory."""
    dataset = []
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            image_files = os.listdir(label_path)[:limit]
            for image_file in image_files:
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, image_size)
                    dataset.append((image, label))
    random.shuffle(dataset)
    return dataset


def process_data(data):
    """Normalize images and encode labels."""
    images, labels = zip(*data)
    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    return images, labels, encoder
