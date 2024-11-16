import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
from datetime import datetime
import csv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/detections'

# Load model
model = YOLO('V:/coding/dev/Python/Gantry/FYP/best.pt')

# Color map for classes
color_map = [
    (124, 252, 0),  # class healthy
    (255, 0, 0),    # class black rot disease
    (0, 124, 252)   # color for other
]

# Create output directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create CSV file for detections
csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'detections.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Timestamp', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

def detect_image(image_path):
    '''Function to perform detection on an uploaded image'''
    frame = cv2.imread(image_path)
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    frame_count = 0

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.2:
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                label = f"{class_name}: {score:.2f}"
                color = color_map[class_id % len(color_map)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                writer.writerow([frame_count, timestamp, class_name, score, x1, y1, x2, y2])

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detection_{frame_count:04d}_{timestamp}.jpg')
    cv2.imwrite(output_path, frame)
    return output_path

def detect_camera():
    '''Function to perform detection on a camera feed'''
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open camera."

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score > 0.2:
                    x1, y1, x2, y2 = map(int, box[:4])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    label = f"{class_name}: {score:.2f}"
                    color = color_map[class_id % len(color_map)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                    writer.writerow([frame_count, timestamp, class_name, score, x1, y1, x2, y2])

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detection_{frame_count:04d}_{timestamp}.jpg')
        cv2.imwrite(output_path, frame)
        frame_count += 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        detection_path = detect_image(file_path)
        return redirect(url_for('uploaded_file', filename=os.path.basename(detection_path)))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('uploaded.html', filename=filename)

@app.route('/video_feed')
def video_feed():
    return Response(detect_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)