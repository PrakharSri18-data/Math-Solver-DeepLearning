# utils.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : This script contains utility functions for preprocessing input images for the CRNN model.
# -------------------------------------------------


# =============================================
# Importing the necessary libraries
# ----------------------------------------
# OpenCV for image processing.
# NumPy for handling data arrays.
# =============================================
import cv2
import numpy as np

# =============================================
# preprocess_image Function
# ----------------------------------------
# This function takes an input image, converts it to grayscale, resizes it to the required dimensions for the CRNN model, normalizes the pixel values, & reshapes it to match the input shape expected by the model.
# It returns the preprocessed image ready for prediction by the CRNN model.
# =============================================
def preprocess_image(image):

    # Converting the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resizing the image to the required dimensions for the CRNN model (256x64)
    img = cv2.resize(gray,(256,64))
    
    # Normalizing the pixel values
    img = img / 255.0
    
    # Reshaping the image to match the input shape expected by the model
    img = img.reshape(1,64,256,1)

    return img