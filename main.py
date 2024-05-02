import numpy as np
import os
import random
import cv2
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
    Conv2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def load_data(directory, max_count):
    data = []
    img_size = 32
    non_chars = ["#", "$", "&", "@"]
    for i in os.listdir(directory):
        if i in non_chars:
            continue
        count = 0
        sub_directory = os.path.join(directory, i)
        for j in os.listdir(sub_directory):
            count += 1
            if count > max_count:
                break
            img = cv2.imread(os.path.join(sub_directory, j), 0)
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, i])
    random.shuffle(data)
    return data


def preprocess_data(data, label_binarizer):
    X = []
    Y = []
    for features, label in data:
        X.append(features)
        Y.append(label)
    Y = label_binarizer.fit_transform(Y)
    X = np.array(X) / 255.0
    X = X.reshape(-1, 32, 32, 1)
    Y = np.array(Y)
    return X, Y


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(35, activation="softmax"))
    return model


def train_model(model, train_X, train_Y, val_X, val_Y, epochs=5, batch_size=32):
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    history = model.fit(
        train_X,
        train_Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_Y),
        verbose=1,
    )
    return history


def predict_letters(img, model, label_binarizer):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y : y + h, x : x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1, 32, 32, 1)
        ypred = model.predict(thresh)
        ypred = label_binarizer.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)
    return letters, image


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    return (cnts, boundingBoxes)


def get_word(letter):
    word = "".join(letter)
    return word


def load_image(predict_text_func):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected.")
        return

    predict_text_func(file_path)


def predict_text(file_path, model, label_binarizer):
    letter, image = predict_letters(file_path, model, label_binarizer)
    word = get_word(letter)

    # Display the prediction result
    result_panel = tk.Toplevel()
    result_panel.title("Prediction Result")
    result_panel.geometry("300x150")

    result_label = tk.Label(
        result_panel, text=f"Predicted Text: {word}", font=("Arial", 12)
    )
    result_label.pack(pady=20)

    # Display the image used for prediction
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))  # Resize the image
    img = Image.fromarray(img)  # Convert to ImageTk format
    img = ImageTk.PhotoImage(image=img)  # Create PhotoImage object

    img_label = tk.Label(result_panel, image=img)
    img_label.image = img
    img_label.pack(padx=10, pady=10)

    result_panel.mainloop()


def main():
    dir_train = "./input/handwritten-characters/Train/"
    dir_val = "./input/handwritten-characters/Validation/"
    train_data = load_data(dir_train, max_count=4000)
    val_data = load_data(dir_val, max_count=1000)

    LB = LabelBinarizer()
    train_X, train_Y = preprocess_data(train_data, LB)
    val_X, val_Y = preprocess_data(val_data, LB)

    num_classes = len(set(train_Y.flatten()))
    input_shape = train_X.shape[1:]

    model = build_model(input_shape, num_classes)
    history = train_model(
        model, train_X, train_Y, val_X, val_Y, epochs=10, batch_size=32
    )

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Training Loss vs Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    load_image(lambda file_path: predict_text(file_path, model, LB))


if __name__ == "__main__":
    main()
