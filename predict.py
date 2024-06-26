import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import time


def extract_and_predict(img_path, cnn_model, encoder):
    """Extract letters from the image and predict them using the trained model."""
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    dilated_image = cv2.dilate(binary_image, None, iterations=2)
    contours = cv2.findContours(
        dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]

    predicted_letters = []
    total_prediction_time = 0

    for contour in contours:
        if cv2.contourArea(contour) > 10:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(
                image, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # Draw bounding box
            roi = gray_image[y : y + h, x : x + w]
            roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_CUBIC)
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = roi.reshape(1, 32, 32, 1)

            start_time = time.time()
            prediction = cnn_model.predict(roi)
            end_time = time.time()
            total_prediction_time += end_time - start_time

            decoded_label = encoder.inverse_transform(prediction)[0]
            predicted_letters.append(decoded_label)

    return "".join(predicted_letters), image, total_prediction_time
