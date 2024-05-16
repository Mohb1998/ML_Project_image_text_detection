from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_cnn(input_shape):
    """Construct a Convolutional Neural Network model."""
    cnn_model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(35, activation="softmax"),
        ]
    )
    return cnn_model


def compile_and_train(
    cnn_model,
    train_images,
    train_labels,
    val_images,
    val_labels,
    epochs=50,
    batch_size=32,
):
    """Compile and train the CNN model."""
    cnn_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    cnn_model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_images, val_labels),
        verbose=1,
    )
