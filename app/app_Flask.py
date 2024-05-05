from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Function to load the pre-trained model and cache it for optimized performance
def load_model():
    model = tf.keras.models.load_model('D:/Image_Classification/models/Flower_Image_Classification_For_12_ClassesH5.h5')
    return model

def preprocess_image(image):
    image = tf.image.resize(image, [150, 150])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def predict_class(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_text='No file uploaded')

    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((150, 150))

    pred = predict_class(np.array(image), model)
    class_names = ['Astilbe', 'Bellflower', 'Black Eyed Susan', 'Calendula', 'California Poppy','Carnation','Common Daisy','Coreopsis','Daffodil','Dandelion','Iris','Magnolia','Rose','Sunflower','Tulip','Water Lily']
    result = class_names[np.argmax(pred)]
    output = 'The image is a ' + result

    return render_template('result.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
