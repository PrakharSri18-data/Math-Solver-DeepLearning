# training/training_crnn.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : Training script for the CRNN model for handwritten text recognition.
# -------------------------------------------------


# =============================================
# Importing the Necessary Libraries
# ----------------------------------------
# Tensorflow for training the CRNN model.
# NumPy for handling data arrays.
# build_crnn function from model_crnn.py to build the model architecture.
# =============================================
import tensorflow as tf
import numpy as np
from model.crnn_model import build_crnn

# Vocabulary for the CRNN model
VOCAB = "0123456789+-*/=()"

# Image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Model parameters
model = build_crnn()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Dummy data for training the model
X = np.random.rand(32,64,256,1)
y = np.random.randint(0,15,(32,10))

# Training the CRNN model
model.fit(
    X,
    y,
    epochs=5,
    batch_size=8
)

# Saving the trained CRNN model
model.save("math_solver.keras")