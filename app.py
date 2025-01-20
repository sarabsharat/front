from flask import Flask, render_template, Response
import os
import cv2
import math
from ultralytics import YOLO
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
app.config['DETECTED_FOLDER'] = 'detected_images'  # Folder to save detected images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the correct path to your trained YOLO model
classNames = ["lock_picking", "holding_gun", "null"]  # Replace with actual class names

@app.route('/')
def index():
    return render_template('Media.html', detected_image=None)

@app.route('/webcam')
def webcam():
    return render_template('Webcam.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    cap.set(3, 640)           # Set frame width
    cap.set(4, 480)           # Set frame height

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection on the frame
        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {conf:.2f}"
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




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
        print(f"File uploaded: {file_path}")  # Debugging statement

        # After file upload, trigger detection
        return detect(file_path)  # Pass file_path to the detection route

@app.route('/detect', methods=['POST'])
def detect(file_path):
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found for detection"}), 400

    # Read image
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Run YOLO detection
    results = model(img)

    # Get the first result
    result = results[0]

    # Optionally, save the result image
    result_path = os.path.join(app.config['DETECTED_FOLDER'], 'detected_image.jpg')
    result.save(result_path)

    # Draw rectangles around detected objects with the new color
    for box in result.boxes.xyxy:  # Assuming boxes are in xyxy format
        x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (64, 110, 142), 2)  # Change color to #406E8E

    # Check confidence level
    if result.boxes.conf[0] < 0.50:
        # Return the image without confidence or class
        return render_template('Media.html', detected_image=result_path)
    
    # Determine threat messages based on detected class
    class_name = None
    threat_message = None
    if len(result.boxes.cls) > 0:  # Check if there are any detected classes
        class_index = int(result.boxes.cls[0].item())  # Get the class index as an integer
        if class_index < len(result.names):  # Ensure the index is within bounds
            class_name = result.names[class_index]  # Get the class name
            if class_name == "holding_gun":
                threat_message = "Threat detected🚨:Someone is holding a gun🔫."
            elif class_name == "lock_picking":
                threat_message = "Threat detected🚨:There's a thief nearby🥷."

    # Return the index page with the detected image and its details
    return render_template('Media.html', detected_image=result_path, class_name=class_name, confidence=result.boxes.conf[0], threat_message=threat_message)

@app.route('/detected_file/<filename>')
def send_detected_file(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
