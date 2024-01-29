import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, Response
import mysql.connector
import re
import cv2
from tensorflow.keras.models import load_model
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_socketio import SocketIO
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
socketio = SocketIO(app)

emotion_model = load_model('emotion_model.h5')
app.secret_key = 'your secret key'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="mini_project"
)

@app.route('/')
def home():
    return render_template('home.html')
movie_data = pd.read_csv('C:\\xampp\\htdocs\\main project\\movie_dataset.csv')
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

@app.route('/admin', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('admin.html'))

@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/admin_dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mydb.cursor(dictionary=True)
        cursor.execute('SELECT * FROM muruka WHERE email = %s AND password = %s', (email, password))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['email'] = account['email'] 
            msg = 'Logged in successfully!'
            return redirect(url_for('dashboard'))
        else:
            msg = 'Incorrect email or password!'
    return render_template('login_register.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and all(key in request.form for key in ['first_name', 'last_name', 'email', 'password', 'confirm_password', 'phone']):
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        phone = request.form['phone']
        show_password = 'show_password' in request.form
        cursor = mydb.cursor(dictionary=True)
        cursor.execute('SELECT * FROM muruka WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif password != confirm_password:
            msg = 'Passwords do not match!'
        elif not email or not password or not first_name or not last_name or not phone:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO muruka (first_name, last_name, email, password, phone) VALUES (%s, %s, %s, %s, %s)',
               (first_name, last_name, email, password, phone))
            mydb.commit()
            msg = 'You have successfully registered!'
            return redirect(url_for('login'))
    return render_template('login_register.html', msg=msg)

@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
  
        email = session['email']

        return render_template('index.html', email=email)
    else:
        return redirect(url_for('login'))
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        email = session.get('email') 

    
        if not (current_password and new_password and confirm_password):
            return "All fields are required."


        if new_password != confirm_password:
            return "New password and confirm password do not match. Please try again."

        
        cursor = mydb.cursor(dictionary=True)
        cursor.execute('SELECT * FROM muruka WHERE email = %s AND password = %s', (email, current_password))
        user = cursor.fetchone()
        if not user:
            return "Current password is incorrect. Please try again."

        
        cursor.execute('UPDATE muruka SET password = %s WHERE email = %s', (new_password, email))
        mydb.commit()

        print(f"Password successfully changed to: {new_password}")

        return "Password successfully changed."

    return render_template('change_password.html')

time_points = []
emotion_labels = []
recommended_movies = []  

emotion_levels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_counts = {emotion_label: 0 for emotion_label in emotion_levels}

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(movie_data['Genre'].fillna(''))
y_train = movie_data['Genre'].fillna('unknown')

def get_movie_recommendations(emotion):
    emotion_genre_mapping = {
        'Angry': 'Action',
        'Disgust': 'Horror',
        'Fear': 'Thriller',
        'Happy': 'Comedy',
        'Sad': 'Drama',
        'Surprise': 'Adventure',
        'Neutral': 'Drama'
    }
    emotion_genre = emotion_genre_mapping.get(emotion, 'Drama')  
    relevant_movies = movie_data[movie_data['Genre'].str.lower().str.contains(emotion_genre.lower())]
    
    # Sort movies based on movie_rating in descending order
    recommended_movies_list = relevant_movies.sort_values(by='movie_rating', ascending=False).head(10)

    # Use the global variable to store recommended movies
    global recommended_movies
    recommended_movies = recommended_movies_list[['MovieName', 'Genre', 'Director', 'movie_rating']]

def update_recommendation():
    global emotion_labels, recommended_movies

    while True:
        # Wait for 1 minute
        time.sleep(60)

        # Check the most frequent emotion during the last minute
        dominant_emotion = max(set(emotion_labels), key=emotion_labels.count)
        print("Dominant Emotion in the last minute:", dominant_emotion)

        # Get movie recommendations based on the dominant emotion
        get_movie_recommendations(dominant_emotion)

        # Clear the emotion labels for the next minute
        emotion_labels.clear()

        # Emit the updated recommendations to the client
        socketio.emit('update_recommendations', recommended_movies.to_dict(orient='records'))

@app.route('/emotion_graph')
def emotion_graph():
    return render_template('emotion_graph.html', recommended_movies=[])

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
    global time_points, emotion_counts

    while True:
        for emotion_label in emotion_labels:
            emotion_counts[emotion_label] += 1

        graph_json = {'image': get_emotion_graph()}
        socketio.emit('update_graph', graph_json)
        time.sleep(1)  # Adjust the sleep time as needed

def get_emotion_graph():
    plt.plot(list(emotion_counts.values()), color='blue', marker='o')
    plt.xticks(range(len(emotion_counts)), list(emotion_counts.keys()))
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.clf()  # Clear the plot for the next iteration
    image_stream.seek(0)

    encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

    return f'data:image/png;base64,{encoded_image}'

def generate_frames():
    global emotion_labels

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    border_color = (0, 255, 0)  # Specify the border color (in BGR format)
    border_thickness = 5  # Specify the border thickness

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray is None:
            print("Gray frame is None")
            continue

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x - border_thickness, y - border_thickness),
                          (x + w + border_thickness, y + h + border_thickness), border_color, border_thickness)

            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = face_roi / 255.0

            emotion_prediction = emotion_model.predict(face_roi)[0]
            emotion_index = np.argmax(emotion_prediction)
            detected_emotion = emotion_levels[emotion_index]

            emotion_labels.append(detected_emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            time_points.append(time.strftime("%H:%M:%S"))

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

if __name__ == '__main__':
    socketio.start_background_task(update_graph)
    socketio.start_background_task(update_recommendation)
    socketio.run(app, debug=True)
 