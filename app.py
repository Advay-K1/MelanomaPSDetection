from flask import Flask, render_template, Response, request, flash, url_for, redirect
import urllib.request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import asarray
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
import tensorflow as tf
import requests
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

predicted = ""

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')

        # IMG_WIDTH = 100
        # IMG_HEIGHT = 75


        # img = tf.keras.preprocessing.image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # img = asarray(img)


        img = cv2.resize(img, (100, 75))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        # zoom = 0.33
        #
        # centerX, centerY = int(IMG_HEIGHT / 2), int(IMG_WIDTH / 2)
        # radiusX, radiusY = int((1 - zoom) * IMG_HEIGHT * 2), int((1 - zoom) * IMG_WIDTH * 2)
        #
        # minX, maxX = centerX - radiusX, centerX + radiusX
        # minY, maxY = centerY - radiusY, centerY + radiusY
        #
        # cropped = img[minX:maxX, minY:maxY]

        print(img.shape)
        #Load model
        keras_model = tf.keras.models.load_model('my_model.h5')
        print (keras_model.summary())
        #test model

        prediction = keras_model.predict(img)

        predicted_class = np.argmax(prediction)

        class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma',
                       'Melanoma', 'Nevus', 'Vascular lesions']
        predicted_class_name = class_names[predicted_class]

        print(f'Predicted class: {predicted_class_name}, Probability: {prediction[0][predicted_class]:.2f}')


        return render_template('index.html', filename=filename, predicted = predicted_class_name)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug = True)