from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import os
#
# # os.en viron['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
app = Flask(__name__)

# Load the trained model from your local file
model = load_model('model.h5')

def predict_image(image_path):
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        return "SAD"
    else:
        return "HAPPY"

@app.route('/')    #Defines the route for the homepage.

def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_predict():
    image_file = request.files.get('image')
    if image_file:
        image_path = os.path.join('static', image_file.filename)
        image_file.save(image_path)

        # Assuming predict_image() returns a string like 'happy' or 'sad'
        prediction = predict_image(image_path)

        # Pass the prediction to the template
        return render_template('index.html', prediction=prediction, image_path=image_path)

    return render_template('index.html', prediction=None, image_path=None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)