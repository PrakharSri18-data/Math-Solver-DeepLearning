# model/crnn_model.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : CRNN Model for handwritten text recognition
# -------------------------------------------------


# =============================================
# Importing the required libraries
# ----------------------------------------
# Tensorflow & Keras for building the CRNN Model
# =============================================
import tensorflow as tf
import keras


# =============================================
# build_crnn function
# ----------------------------------------
# This function builds the CRNN model architecture for handwritten text recognition.
# It takes the image height, image width, & vocbulary size as input parameters & returns the compiled CRNN model.
# =============================================
def build_crnn(img_height=64, img_width=256, vocab_size=16):

    # Input layer
    input_img = keras.layers.Input(shape=(img_height, img_width, 1))

    # Convolution layers
    x = keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(input_img)
    x = keras.layers.MaxPooling2D((2,2))(x)

    x = keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)

    x = keras.layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)

    # Reshape for RNN
    new_shape = ((img_width // 8), (img_height // 8) * 256)
    x = keras.layers.Reshape(target_shape=new_shape)(x)

    # Dense layer
    x = keras.layers.Dense(64, activation="relu")(x)

    # Bidirectional LSTM layers
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(x)

    # Output layer
    output = keras.layers.Dense(vocab_size + 1, activation="softmax")(x)

    model = keras.Model(inputs=input_img, outputs=output)

    return model