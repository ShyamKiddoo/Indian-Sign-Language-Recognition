import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import pyttsx3
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers

# Load the saved model from file
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Function to convert text to audio asynchronously
def text_to_audio(label):
    engine.say(label)
    engine.runAndWait()

# Functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# For webcam input
cap = cv2.VideoCapture(0)
frame_count = 0 
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        frame_count += 1
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Draw the landmarks
                for landmark in landmark_list:
                    cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Ensure the input shape is correct for LSTM
                lstm_input = np.expand_dims([pre_processed_landmark_list], axis=-1)

                # Predict the sign language
                predictions = model.predict(lstm_input, verbose=0)
                # Get the predicted class for each sample
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                print(label)
                print("------------------------")
                
                # Convert label text to audio asynchronously
                threading.Thread(target=text_to_audio, args=(label,)).start()

        # Output image
        cv2.imshow('Indian sign language detector', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("Number of frames captured:", frame_count)
print("Frame rate (fps):", frame_rate)

cap.release()
cv2.destroyAllWindows()