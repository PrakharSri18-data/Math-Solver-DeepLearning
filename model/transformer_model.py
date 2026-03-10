# model/transformer_model.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : Transformer Model for handwritten text recognition
# -------------------------------------------------


# =============================================
# Difference between CRNN & Transformer models
# ----------------------------------------
# 1. Architecture:
#   - CRNN: Combines Convolutional layers for feature extraction & Recurrent layers (LSTM/GRU) for sequence modeling.
#   - Transformer: Uses self-attention mechanisms to capture long-range dependencies without recurrence.
# 2. Parallelization:
#   - CRNN: Sequential processing due to recurrent layers, making it less efficient for long sequences.
#   - Transformer: Allows for parallel processing of input sequences, improving training efficiency.
# 3. Performance:
#   - CRNN: May struggle with long sequences due to vanishing gradients in RNNs.
#   - Transformer: Handles long sequences effectively with self-attention, often achieving better performance.
# 4. Use Cases:
#   - CRNN: Suitable for tasks with shorter sequences or where spatial features are crucial.
#   - Transformer: Preferred for tasks with longer sequences or where capturing global context is important.
# =============================================


# =============================================
# Importing the Necessary Libraries
# ----------------------------------------
# Tensorflow & Keras for building the transformer model.
# =============================================
import tensorflow as tf
import keras


# =============================================
# build_transformer_model Function
# ----------------------------------------
# This function builds the transformer model architecture for handwritten text recognition.
# It takes the image height, image width, & vocabulary size as input parameters & returns the compiled transformer model.
# =============================================
def build_transformer_model(img_height=64, img_width=256, vocab_size=16):

    inputs = keras.layers.Input(shape=(img_height, img_width, 1))

    # CNN backbone
    x = keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
    x = keras.layers.MaxPooling2D((2,2))(x)

    x = keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)

    # Convert feature map to sequence
    new_shape = ((img_width // 4), (img_height // 4) * 128)
    x = keras.layers.Reshape(target_shape=new_shape)(x)

    # Transformer Encoder
    attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)

    x = keras.layers.Add()([x, attn])
    x = keras.layers.LayerNormalization()(x)

    x = keras.layers.Dense(256, activation="relu")(x)

    outputs = keras.layers.Dense(vocab_size + 1, activation="softmax")(x)
    
    # Create the model
    model = keras.Model(inputs, outputs)

    return model