from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the pretrained InceptionV3 model and the trained LSTM model
def load_pretrained_model():
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    return model

pretrained_model = load_pretrained_model()
model = load_model('violence_detection_model_sgd.h5')

def extract_video_frames_for_prediction(video_path, sequence_length=16, image_width=299, image_height=299):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (image_height, image_width))
        frames_list.append(resized_frame)

    video_reader.release()
    return frames_list

def extract_frame_features_for_prediction(frame, pretrained_model):
    img = np.expand_dims(frame, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    feature_vector = pretrained_model.predict(img, verbose=0)
    return feature_vector

def predict_violence_in_video(video_path, pretrained_model, model):
    frames = extract_video_frames_for_prediction(video_path)
    frames_features = [extract_frame_features_for_prediction(frame, pretrained_model) for frame in frames]
    frames_features = np.array(frames_features).reshape((1, 16, 2048))

    prediction = model.predict(frames_features)
    return prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = 'uploaded_video.mp4'
        file.save(file_path)
        prediction = predict_violence_in_video(file_path, pretrained_model, model)
        result = 'Violence detected' if prediction > 0.5 else 'No violence detected'
        return render_template('result.html', result=result)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
