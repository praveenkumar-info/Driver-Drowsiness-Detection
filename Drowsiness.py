from unittest import result
import cv2
from cv2 import VideoCapture
from lightgbm import cv
from matplotlib import image
import mediapipe as mp
from nbformat import read
from winsound import SND_ALIAS, PlaySound 
import winsound
import face_recognition
from datetime import datetime
import pywhatkit
import pyautogui
import datetime

known_image = face_recognition.load_image_file("C:\\Users\\rithi\\Downloads\\WhatsApp Image 2023-12-14 at 11.37.31 AM.jpeg")
biden_encoding = face_recognition.face_encodings(known_image)[0]

unknown_face_count = 0
count_eye_closed = 0

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
my_face_mesh  = mp.solutions.face_mesh
face = my_face_mesh.FaceMesh(max_num_faces = 4)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS,100)
while True:
    cap,frame = video.read()
    height_frame , width_frame,_ = frame.shape
    # frame=cv2.resize(frame,(1000,700))
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   
   
    # ====================================================================================================
    cap1,frame1 = video.read()
   
   
   
       
    unknown_location = face_recognition.face_locations(frame1)
   
    unknown_encoding = face_recognition.face_encodings(frame1,unknown_location)
    for face_new in unknown_encoding:
       
        results_new = face_recognition.compare_faces([biden_encoding],face_new)
       
        match = str(results_new)
       
       
       
        # compare two faces
        if match == "[True]" :
            unknown_face_count = 0
            name = "Drowsiness detection"
           
        if match == "[False]":
            name = "unknown person locked"
       
        if name == "unknown person locked":
            unknown_face_count += 1
            # print(unknown_face_count)
       
        if unknown_face_count > 5 :
            date = datetime.datetime.now()
            p = str(date.time())

            m = p.split(":")
         

            pywhatkit.sendwhatmsg("+919003987083",
                                "unknown person detected",
                                int(m[0]),  int(m[1])+1)
            unknown_face_count = 0
            pyautogui.press("enter")
       
             
       
        cv2.putText(frame,name,(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3,cv2.LINE_AA)
   
   
   
    results = face.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
           
            #  taking coordinates of eyes points in face x and y axis
           
            e1_r = round(results.multi_face_landmarks[0].landmark[385].y*height_frame)
            e2_r = round(results.multi_face_landmarks[0].landmark[380].y*height_frame)
           
            e3_r = round(results.multi_face_landmarks[0].landmark[386].y*height_frame)
            e4_r = round(results.multi_face_landmarks[0].landmark[374].y*height_frame)
           
            e5_r = round(results.multi_face_landmarks[0].landmark[387].y*height_frame)
            e6_r = round(results.multi_face_landmarks[0].landmark[373].y*height_frame)
           
            e1_l = round(results.multi_face_landmarks[0].landmark[159].y*height_frame)
            e2_l = round(results.multi_face_landmarks[0].landmark[145].y*height_frame)
           
            e3_l = round(results.multi_face_landmarks[0].landmark[159].y*height_frame)
            e4_l = round(results.multi_face_landmarks[0].landmark[145].y*height_frame)
           
            e5_l = round(results.multi_face_landmarks[0].landmark[158].y*height_frame)
            e6_l = round(results.multi_face_landmarks[0].landmark[153].y*height_frame)
           
            # for checking the points in face where it is
           
            # e1_c = round(results.multi_face_landmarks[0].landmark[385].x*width_frame)
            # e2_c = round(results.multi_face_landmarks[0].landmark[385].y*height_frame)
           
            # e3_c = round(results.multi_face_landmarks[0].landmark[380].x*width_frame)
            # e4_c = round(results.multi_face_landmarks[0].landmark[380].y*height_frame)
           
            # cv2.circle(frame,(e1_c,e2_c),radius=6,color=(180,20,40),thickness=-1)
            # cv2.circle(frame,(e3_c,e4_c),radius=6,color=(180,20,40),thickness=-1)
           
            mp_draw.draw_landmarks(frame,face_landmarks,my_face_mesh.FACEMESH_CONTOURS,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(color=(255,255,255),thickness=1, circle_radius=1))
           
            # condition eyes closed
            if (e1_r - e2_r) in [i for i in range(-8,9)] and (
                    
                    e3_r - e4_r) in [i for i in range(-8,9)] and (e5_r - e6_r) in [i for i in range(-7,8)] and (e1_l - e2_l) in [i for i in range(-8,9)] and (e3_l - e4_l) in [i for i in range(-8,10)] and (e5_l - e6_l) in [i for i in range(-7,9)]:
                cv2.putText(frame,"eyes closed",(100,450),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3,cv2.LINE_AA)
                count_eye_closed += 1
               
            else:
                count_eye_closed = 0
                cv2.putText(frame,"eyes open",(100,450),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3,cv2.LINE_AA)
               
            if count_eye_closed > 5:
                winsound.PlaySound("C:\\Users\\rithi\\Downloads\\mixkit-alarm-tone-996.wav",SND_ALIAS)
                count_eye_closed = 0
                date = datetime.datetime.now()
                p = str(date.time())
                m = p.split(":")
                pywhatkit.sendwhatmsg("+919003987083",
                                    "Driver sleepy",
                                    int(m[0]),  int(m[1])+1)
                count_eye_closed = 0
                pyautogui.press("enter")

           
    cv2.imshow("image",frame)
    key =  cv2.waitKey(1)
    if key == 81:
        break
   
video.release()
cv2.destroyAllWindows()
#  for right eye
# up
# 385 , 386 , 387
# down
#  380 ,374, 373



# =====================

#  for left eye
# up
# 160,159,158
# down
#  144,145,153
