import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence= 0.3)

DATA_DIR = r'C:\Users\raksh\Desktop\Detecing hand signs ML project'
valid_dirs = [str(i) for i in range(26)]

data = []
labels = []

for dir_ in valid_dirs:
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        full_img_path = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(full_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # len(hand_landmarks.landmark) gives the number of points
                # hand_landmarks.landmark gives a list of all points [xyz xyz....]
                for i in range(len(hand_landmarks.landmark)):
                    x_ = hand_landmarks.landmark[i].x #x coordinae of ith point of the image
                    y_ = hand_landmarks.landmark[i].y
                    #we need only the x and y coordinates because z is just the distance from the camera
                    data_aux.append(x_)
                    data_aux.append(y_)

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels},f)
f.close()
