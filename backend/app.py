from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Load the trained model from your local file
model = load_model('content/img_classification/models/imageclassifier_happy-sad.h5')

def predict_image(image_path):
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        return "SAD"
    else:
        return "HAPPY"

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join('static', image_file.filename)
            image_file.save(image_path)
            prediction = predict_image(image_path)
            return render_template('index.html', prediction=prediction, image_path=image_path)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
