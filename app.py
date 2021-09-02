import cv2
from flask import Flask, render_template, Response, request
import numpy as np
import os
import tensorflow as tf

global switch, face
face=0
switch=1

# Flask Setup
app = Flask(__name__)
camera = cv2.VideoCapture(0)
model_path = os.path.join('static','models','emotion_model_full.h5')
emotion_model = tf.keras.models.load_model(model_path)

#Functions

def emotion_prediction(frame):
    emotion_dict = {0: "Angry",
                    1: "Disgusted",
                    2: "Fearful",
                    3: "Happy",
                    4: "Neutral",
                    5: "Sad",
                    6: "Surprised"}
    bounding_box_path = os.path.join('static','xml','haarcascade_frontalface_default.xml')
    # Use haar cascade to draw bounding box around face
    bounding_box = cv2.CascadeClassifier(bounding_box_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.flip(cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA),1)
    
    return frame

def gen_frames():  # generate frame by frame from camera
    while True:
        success, reversed_frame = camera.read()
        frame = cv2.flip(reversed_frame, 1)
        if success:
            if(face):                
                frame = emotion_prediction(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass

# Flask Routes
@app.route("/")
def index():
    return render_template('indexmer.html')

@app.route("/sources")  
def sources():
  return render_template("sources.html")    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('face') == 'Detect Emotion':
            global face
            face=not face 
            if(face):
                pass
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch=1
             
    elif request.method=='GET':
        return render_template('indexmer.html')
    return render_template('indexmer.html')  
   
if __name__ == "__main__":
    app.run()

camera.release()
cv2.destroyAllWindows()  