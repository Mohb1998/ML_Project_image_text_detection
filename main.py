import joblib
import numpy as np
import os
import random
import cv2
import imutils
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
    Conv2D,
)
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

epochs = 50
batch_size = 32
prediction_time = 0


def load_data(directory, max_count):
    data = []
    for i in os.listdir(directory):
        if i in ["#", "$", "&", "@"]:
            continue
        count = 0
        sub_directory = os.path.join(directory, i)
        for j in os.listdir(sub_directory):
            count += 1
            if count > max_count:
                break
            img = cv2.imread(os.path.join(sub_directory, j), 0)
            img = cv2.resize(img, (32, 32))
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


def build_model(input_shape):
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
    model.add(Dense(35, activation="softmax")) #actually gives the prediction 
    return model


def train_model(model, train_X, train_Y, val_X, val_Y):
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
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    global prediction_time
    prediction_time = 0

    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1, 32, 32, 1)

        start_time = time.time()
        ypred = model.predict(thresh)
        end_time = time.time()
        prediction_time += end_time - start_time

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


def load_image(model, label_binarizer): #GUI 
    root = tk.Tk()
    root.title("Welcome")
    root.geometry("600x400")

    welcome_label = tk.Label(
        root,
        text="Hi, welcome to our image to text program",
        font=("Arial", 14),
        pady=20,
    )
    welcome_label.pack()

    next_button = tk.Button(
        root,
        text="Next",
        font=("Arial", 12),
        padx=10,
        pady=5,
        command=lambda: select_image_page(root, model, label_binarizer),
    )
    next_button.pack()

    root.mainloop()


def select_image_page(root, model, label_binarizer): # GUI 
    root.destroy()

    root = tk.Tk()
    root.title("Select Image")
    root.geometry("600x400")

    select_label = tk.Label(
        root,
        text="Please select the image file that you would like to convert to text",
        font=("Arial", 14),
        pady=20,
    )
    select_label.pack()

    select_button = tk.Button(
        root,
        text="Select Image",
        font=("Arial", 12),
        padx=10,
        pady=5,
        command=lambda: load_and_predict(root, model, label_binarizer),
    )
    select_button.pack()

    root.mainloop()


def load_and_predict(root, model, label_binarizer): #GUI 
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected.")
        return

    predict_text(file_path, model, label_binarizer)


def predict_text(file_path, model, label_binarizer):
    letters, _ = predict_letters(file_path, model, label_binarizer)
    word = "".join(letters)

    # Create the 'results' folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Append the predicted text to the existing file or create a new file if it doesn't exist
    result_file_path = os.path.join("results", "predicted_text.txt")
    with open(result_file_path, "a") as file:
        file.write(f"Predicted Text: {word}\n")
        prediction_time_ms = round(prediction_time * 1000)  # Convert to milliseconds
        file.write(f"Total Prediction Time: {prediction_time_ms} ms\n")
        file.write("\n")  # Add a blank line for separation

    result_panel = tk.Toplevel()
    result_panel.title("Prediction Result")
    result_panel.geometry("600x500")

    result_label = tk.Label(
        result_panel, text=f"Predicted Text: {word}", font=("Arial", 12), pady=20
    )
    result_label.pack()

    prediction_time_ms = round(prediction_time * 1000)  # Convert to milliseconds
    prediction_time_label = tk.Label(
        result_panel,
        text=f"Total Prediction Time : {prediction_time_ms} ms",
        font=("Arial", 12),
        pady=40,
    )
    prediction_time_label.pack()

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    img_label = tk.Label(result_panel, image=img)
    img_label.image = img
    img_label.pack(padx=10, pady=10)

    exit_button = tk.Button(
        result_panel,
        text="Exit",
        font=("Arial", 12),
        padx=10,
        pady=5,
        command=result_panel.destroy,
    )
    exit_button.pack(side="left", padx=10)

    try_again_button = tk.Button(
        result_panel,
        text="Try Again",
        font=("Arial", 12),
        padx=10,
        pady=5,
        command=lambda: select_image_page(result_panel, model, label_binarizer),
    )
    try_again_button.pack(side="right", padx=10)

    result_panel.mainloop()


def main():
    dir_train = "./input/handwritten-characters/Train/"
    dir_val = "./input/handwritten-characters/Validation/"
    if os.path.exists("./model.h5"): 
        model = load_model("./model.h5")
        LB = joblib.load("./label_binarizer.pkl")

    else:
        print("not helo")
        train_data = load_data(dir_train, max_count=4000)
        val_data = load_data(dir_val, max_count=1000)

        train_labels = [label for _, label in train_data]
        LB = LabelBinarizer()
        LB.fit(train_labels)
        joblib.dump(LB, "label_binarizer.pkl")

        train_X, train_Y = preprocess_data(train_data, LB)
        val_X, val_Y = preprocess_data(val_data, LB)

        input_shape = train_X.shape[1:]

        model = build_model(input_shape)
        train_model(model, train_X, train_Y, val_X, val_Y)
        model.save("model.h5")

    load_image(model, LB)


if __name__ == "__main__":
    main()
