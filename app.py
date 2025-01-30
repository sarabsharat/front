from flask import Flask, render_template, Response
import os
import cv2
import math
from ultralytics import YOLO
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
cap = None
webcam_running = False

app.config['UPLOAD_FOLDER'] = 'uploads' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['DETECTED_FOLDER'] = 'detected_images'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

model = YOLO('best.pt')  
classNames = ["lock_picking", "holding_gun", "null"]  

@app.route('/')
def index():
    return render_template('Media.html', detected_image=None)

@app.route('/webcam')
def webcam():
    return render_template('Webcam.html')

def generate_frames():
    global cap
    global webcam_running
    cap = cv2.VideoCapture(0)  
    cap.set(3, 640)           
    cap.set(4, 480)          

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                conf = box.conf[0]
                if conf >= 0.5:  # Only draw box if confidence is 50% or higher
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{classNames[int(box.cls[0])]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (142, 110, 64), 4) 
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (142, 110, 64), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global cap, webcam_running
    if not webcam_running:
        cap = cv2.VideoCapture(0)  
        cap.set(3, 640)           
        cap.set(4, 480)          
        webcam_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global cap, webcam_running
    if cap is not None and cap.isOpened():
        cap.release()  
        webcam_running = True
        return "Webcam stopped"
    else:
        return "Webcam was not started"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File uploaded: {file_path}") 

        return detect(file_path)  

@app.route('/detect', methods=['POST'])
def detect(file_path):
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found for detection"}), 400

    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Failed to load image"}), 400

    results = model(img)
    result = results[0]

    result_path = os.path.join(app.config['DETECTED_FOLDER'], 'detected_image.jpg')
    result.save(result_path)

    class_name = None
    threat_message = None

    if len(result.boxes) == 0:  # Check if there are no detected boxes
        threat_message = "No threats detected."
    else:
        confidence = result.boxes.conf[0].item()  
        if confidence < 0.5:  # If confidence is less than 50%
            return render_template('Media.html', detected_image=result_path, class_name=None, confidence=None, threat_message="No threats detected.")

        class_index = int(result.boxes.cls[0].item())  
        if class_index < len(result.names): 
            class_name = result.names[class_index]
            if class_name == "holding_gun":
                threat_message = "Threat detected🚨: Someone is holding a gun🔫."
            elif class_name == "lock_picking":
                threat_message = "Threat detected🚨: There's a thief nearby🥷."

    return render_template('Media.html', detected_image=result_path, class_name=class_name, confidence=confidence if class_name else None, threat_message=threat_message)

@app.route('/detected_file/<filename>')
def send_detected_file(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
