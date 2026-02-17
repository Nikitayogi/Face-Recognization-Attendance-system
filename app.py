from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import face_recognition
import csv
from datetime import datetime
import threading

app = Flask(__name__)

camera = None
running = False

# ===============================
# Load Known Faces
# ===============================
path = os.path.join(os.getcwd(), "static/faces")

images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

known_face_encodings = findEncodings(images)
known_face_names = classNames


# ===============================
# Mark Attendance
# ===============================
def markAttendance(name):
    file_exists = os.path.isfile('attendance.csv')

    with open('attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        # Prevent duplicate same day
        with open('attendance.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                if len(row) > 1 and row[0] == name and row[1] == date:
                    return

        writer.writerow([name, date, time])


# ===============================
# Video Streaming Function
# ===============================
def generate_frames():
    global camera, running

    while running:
        success, frame = camera.read()
        if not success:
            break

        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(known_face_encodings, encodeFace)
            faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = known_face_names[matchIndex].upper()
                markAttendance(name)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ===============================
# Routes
# ===============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    global camera, running
    if not running:
        camera = cv2.VideoCapture(0)
        running = True
    return render_template('index.html')


@app.route('/stop')
def stop():
    global running, camera
    running = False
    if camera:
        camera.release()
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
