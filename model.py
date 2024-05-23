from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_cnn(input_shape):
    """Construct a Convolutional Neural Network model."""
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(35, activation='softmax'))
    model.summary()
    return model


def compile_and_train(
    cnn_model,
    train_images,
    train_labels,
    val_images,
    val_labels,
    epochs=16,
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
