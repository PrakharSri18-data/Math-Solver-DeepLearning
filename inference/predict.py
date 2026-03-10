# inference/predict.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : Inference script for predicting handwritten equations using the trained CRNN model.
# -------------------------------------------------


# =============================================
# Importing the Necessary Libraries
# ----------------------------------------
# Tensorflow for loading the trained model & making predictions.
# NumPy for handling data arrays.
# =============================================
import tensorflow as tf
import numpy as np

# Vocabulary for the CRNN model
VOCAB = "0123456789+-*/=()"

# Mapping the numeric predictions back to characters
num_to_char = {i:c for i,c in enumerate(VOCAB)}

# Loading the trained CRNN model
model = tf.keras.models.load_model(
    "math_solver_crnn.keras",
    compile=False
)


# =============================================
# decode_prediction Function
# ----------------------------------------
# This function decodes the model's predictions using CTC decoding to convert the predicted sequences of numbers back into readable equations.
# It takes the model's raw predictions as input & returns the decoded equation as a string.
# =============================================
def decode_prediction(pred):

    # CTC decoding to convert the predicted sequences of numbers back into readable equations
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    result = ""
    
    # Mapping the numeric predictions back to characters using the num_to_char dictionary
    for x in decoded.numpy()[0]:
        if x != -1:
            result += num_to_char[x]
    return result


# =============================================
# predict_equation Function
# ----------------------------------------
# This function takes an input image, preprocesses it, feeds it to the trained CRNN model for prediction, & decodes the output to return the predicted equation as a string.
# It takes the input image as a parameter & returns the predicted equation as a string.
# =============================================
def predict_equation(image):

    # Predicting the equation from the input image using the trained CRNN model
    pred = model.predict(image)
    equation = decode_prediction(pred)
    return equation