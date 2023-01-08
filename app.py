################################# IMPORTING REQUIRED PACKAGES######################################

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time
import streamlit as st
from PIL import ImageFont, ImageDraw, Image


################################### BUILDING KEYPOINTS USING MEDIAPIPE HOLISTICS###################

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
modelF= keras.models.load_model('rec_0.h5')


################################# CAPTURING REAL TIME VIDEO ###########################################

cap = cv2.VideoCapture(0)
fontpath = r'.\Tamil003.ttf' 
font = ImageFont.truetype(fontpath, 32)

############################### MAPPING THE PREDICTION IN ENGLISH TO TAMIL ############################################

d = {'hello':'வணக்கம்', 'thanks':'நன்றி', 'i love you':'நான் உன்னை காதலிக்கிறேன்', 'stop':'நிறுத்து', 
'yes':'ஆம்','see':'பார்க்க','walk':'நடை','argue':'வாதிடு','good':'நல்ல'}

#################################### OPEN CAMERA ######################################################################

class OpenCamera ():
    
    def __init__(self) -> None :
        self.sequence = []
        self.sentence = []
        self.threshold = 0.4
        self.actions = np.array(['hello', 'thanks', 'i love you', 'stop', 'please', 'walk', 'argue', 'yes', 'see', 'good'])

############################### DETECTING THE SIGN BY PASSING FRAMES TO THE #############################################

    def mediapipe_detection(self,image, model):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False                 
        self.results = model.process(image)               
        self.image.flags.writeable = True                   
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.image, self.results

############################## CREATING MARKING ON THE PALM TO GET KEY POINTS TO DETERMINE THE SIGN WITH GOOD ACCURACY ############

    def draw_styled_landmarks(self,image, results):
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

########################## EXTRACTING KEYPOINTS######################################################################################
    
    def extract_keypoints(self, results):
        self.key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        self.key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        self.lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        self.rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([self.key1, self.key2, self.lh, self.rh])


    def recv(self):
        ctime = 0
        ptime = 0
        # st.video(cap.read())
        frame_window = st.image([])
        while cap.isOpened():
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(id, cx, cy)
                        if id==8:
                            cv2.circle(img, (cx, cy), 10, (100, 200, 250), cv2.FILLED)
                    mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                image, results = self.mediapipe_detection(img,holistic)
                self.draw_styled_landmarks(image, results)

############################## 2. Prediction logic ####################################################################
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30:
                    res = modelF.predict(np.expand_dims(self.sequence, axis=0))[0]
                    

##################################3. Viz logic ###########################################################################

                    if res[np.argmax(res)] > self.threshold: 
                        if len(self.sentence) > 0: 
                            if self.actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(self.actions[np.argmax(res)])
                        else:
                            self.sentence.append(self.actions[np.argmax(res)])
                        
                    if len(self.sentence) > 1: 
                        self.sentence = self.sentence[-1:]

                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    b,g,r,a = 0,255,0,0 #white

                    draw.text((3,30),  d[self.sentence[0]], font = font, fill = ((b, g, r, a)))
                    res_img = np.array(img_pil)
                    
                    img[0:720, 0:1280] = res_img
                    
                    ctime = time.time()
                    fps = 1/(ctime-ptime)
                    ptime = ctime
    
                    cv2.putText(img, str(int(fps)), (10 ,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    frame_window.image(frame)
        
       
                    
                    
                    
                    
                   
            

oc = OpenCamera()
oc.recv()