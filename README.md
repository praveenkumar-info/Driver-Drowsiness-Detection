# Driver Drowsiness Detection using Python
This Python script implements a Driver Drowsiness Detection System using computer vision, face recognition, and machine learning techniques. The system utilizes the face_recognition library to compare faces, OpenCV for facial landmark detection, and MediaPipe for face mesh analysis.

**Features:**
1. Real-time facial landmark detection and eye tracking to monitor driver behavior.
2. Face recognition to identify known individuals and trigger alerts for unknown persons.
3. Alarm sound and automated message alerts through WhatsApp using PyWhatKit when signs of drowsiness are detected. 

**Dependencies:**

1. face_recognition
2. cv2 (OpenCV)
3. mediapipe
4. pywhatkit
5. winsound
   
**How It Works:**

1. The system compares detected faces with a pre-loaded known face using face_recognition.
2. Real-time eye tracking is performed using facial landmarks obtained with MediaPipe.
3. An alarm is triggered and a WhatsApp message is sent if the driver's eyes remain closed for an extended period, indicating potential drowsiness.
4. Face recognition ensures alerts are only sent for unknown persons.
   
**Usage:**

1. Run the script with a webcam to monitor the driver's face in real-time.
2. Ensure dependencies are installed using pip install -r requirements.txt.
3. Customize the alarm sound and recipient's WhatsApp number in the script.
   
Feel free to contribute and enhance the functionality of this Driver Drowsiness Detection System!.
