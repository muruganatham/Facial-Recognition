import numpy as np
from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_socketio import SocketIO
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
socketio = SocketIO(app)

emotion_model = load_model('emotion_model.h5')

# Initialize global variables for emotion data
time_points = []
emotion_labels = []

# Emotion levels and corresponding colors
emotion_levels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'purple', 'gray']

emotion_counts = {label: 0 for label in emotion_levels}

@app.route('/')
def index():
    return render_template('emotion_graph.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def update_graph():
    global time_points, emotion_labels, emotion_counts

    while True:
        # Create a line graph of emotion frequency
        plt.plot(list(emotion_counts.values()), color='blue', marker='o')
        plt.xticks(range(len(emotion_counts)), list(emotion_counts.keys()))
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')

        # Convert the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        plt.clf()  # Clear the plot for the next iteration
        image_stream.seek(0)

        # Convert the BytesIO object to base64 for embedding in HTML
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

        # Update the emotion counts for the next iteration
        for emotion_label in emotion_labels:
            emotion_counts[emotion_label] += 1

        # Convert the figure to JSON
        graph_json = {'image': f'data:image/png;base64,{encoded_image}'}
        socketio.emit('update_graph', graph_json)
        time.sleep(1)  # Adjust the sleep time as needed

def generate_frames():
    global time_points, emotion_labels

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    border_color = (0, 255, 0)  # Specify the border color (in BGR format)
    border_thickness = 5  # Specify the border thickness

    while True:
        ret, frame = cap.read()

        # Print statements for debugging
        if not ret:
            print("Failed to capture frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # More print statements for debugging
        if gray is None:
            print("Gray frame is None")
            continue

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a border around the face
            cv2.rectangle(frame, (x - border_thickness, y - border_thickness),
                          (x + w + border_thickness, y + h + border_thickness), border_color, border_thickness)

            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = face_roi / 255.0

            emotion_prediction = emotion_model.predict(face_roi)[0]
            emotion_index = np.argmax(emotion_prediction)
            emotion_label = emotion_levels[emotion_index]

            # Append the emotion label to the global variable
            emotion_labels.append(emotion_label)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw emotion label on the frame
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Create a mirrored version of the frame
        mirrored_frame = cv2.flip(frame, 1)

        # Concatenate the original and mirrored frames horizontally
        side_by_side = np.concatenate((frame, mirrored_frame), axis=1)

        ret, jpeg = cv2.imencode('.jpg', side_by_side)
        frame_bytes = jpeg.tobytes()

        # Update global variables with the current time
        time_points.append(time.strftime("%H:%M:%S"))

        # Yield the frame for video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


if __name__ == '__main__':
    socketio.start_background_task(update_graph)
    socketio.run(app, debug=True)
