import os
import joblib
from tensorflow.keras.models import load_model
from data_processing import fetch_data, process_data
from model import create_cnn, compile_and_train
from gui import initialize_interface


def main():
    train_directory = "input/handwritten-characters/Train/"
    validation_directory = "input/handwritten-characters/Validation/"

    if os.path.exists("ML_Project_image_text_detection/trained_model.h5") and os.path.exists("ML_Project_image_text_detection/label_encoder.pkl"):
        cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")
        label_encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
    else:
        train_dataset = fetch_data(train_directory)
        validation_dataset = fetch_data(validation_directory)

        train_images, train_labels, label_encoder = process_data(train_dataset)
        val_images, val_labels, _ = process_data(validation_dataset)

        cnn_model = create_cnn(train_images.shape[1:])
        compile_and_train(cnn_model, train_images, train_labels, val_images, val_labels)

        cnn_model.save("trained_model.h5")
        joblib.dump(label_encoder, "label_encoder.pkl")

    initialize_interface(cnn_model, label_encoder)


if __name__ == "__main__":
    main()
