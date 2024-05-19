import os
import random
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
random.seed(42)

def fetch_data(directory_path, limit=4000, image_size=(32, 32)):
    """Fetch images and corresponding labels from the directory."""
    dataset = []
    for label in os.listdir(directory_path):
        if label in ["#", "$", "&", "@"]:
            continue
        count = 0
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                count += 1
                if count > limit:
                    break
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
