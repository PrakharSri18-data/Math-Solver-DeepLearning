# webcam_app.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : This script captures video from the webcam, preprocesses the frames, uses the trained CRNN model to predict the equations in real-time, & displays the predicted equations on the video feed.
# -------------------------------------------------


# =============================================
# Importing the necessary libraries
# ----------------------------------------
# OpenCV for video capture and image processing.
# preprocess_image function from utils.py for preprocessing the input frames.
# predict_equation function from inference/predict.py for making predictions using the trained CRNN model.
# =============================================
import cv2
from utils import preprocess_image
from inference.predict import predict_equation

# Starting video capture from the webcam
cap = cv2.VideoCapture(0)

# Main loop for capturing and processing video frames
while True:
    ret,frame = cap.read()
    img = preprocess_image(frame)
    eq = predict_equation(img)
    cv2.putText(frame,eq,(50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)
    cv2.imshow("Math Solver",frame)
    
    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Releasing the video capture and closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()