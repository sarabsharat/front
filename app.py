from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
app.config['DETECTED_FOLDER'] = 'detected_images'  # Folder to save detected images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the correct path to your trained YOLO model

@app.route('/')
def index():
    return render_template('Media.html', detected_image=None)

@app.route('/webcam')
def webcam():
    return render_template('Webcam.html')

#صثلاؤشة رهي
def generate_frames():
    camera = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = camera.read()  # Read a frame from the webcam
        if not success:
            break
        
        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()  # Annotate the frame with detection results
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield the frame

@app.route('/')
def index():
    return render_template('index.html')  # Render the main page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#gggggggggggggggggggggggggggggggggggggggggggg

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

    # Extract confidence and detected classes
    confidence = result.boxes.conf[0].item()  # Assuming you want the first detection's confidence
    detected_classes = result.names[result.boxes.cls[0].item()]  # Get the class name

    # Optionally, save the result image
    result_path = os.path.join(app.config['DETECTED_FOLDER'], 'detected_image.jpg')
    result.save(result_path)

    # Return the index page with the detected image and additional data
    return jsonify({
        "message": "Detection successful",
        "image_path": result_path,
        "confidence": confidence,
        "detected_classes": detected_classes
    })

@app.route('/detected_file/<filename>')
def send_detected_file(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

#hsgdsa  if __name__ == '__main__':   app.run(host='0.0.0.0', port=5000)

