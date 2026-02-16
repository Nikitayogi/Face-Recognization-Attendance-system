from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ------------------ Load Known Faces ------------------ #
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# ------------------ Attendance Function ------------------ #
def markAttendance(name):
    try:
        df = pd.read_csv('Attendance.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Time'])
    if name not in df['Name'].values:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        df = pd.concat([df, pd.DataFrame([[name, dtString]], columns=['Name','Time'])], ignore_index=True)
        df.to_csv('Attendance.csv', index=False)

# ------------------ Video Capture ------------------ #
cap = cv2.VideoCapture(0)
capture_active = True  # Global flag to control video capture

def gen_frames():
    global capture_active
    while capture_active:
        success, img = cap.read()
        if not success:
            break
        else:
            # Resize for faster recognition
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    markAttendance(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale back
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------ Flask Routes ------------------ #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global capture_active, cap
    capture_active = False
    cap.release()
    cv2.destroyAllWindows()
    return "Attendance Complete"

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    app.run(debug=True)
