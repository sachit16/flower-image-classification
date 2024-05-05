import streamlit as st  # Importing Streamlit library for creating the app interface
import tensorflow as tf  # Importing TensorFlow for using deep learning functionalities
import numpy as np  # Importing NumPy for numerical operations
from PIL import Image  # Importing Image module from PIL library for image processing

# Function to load the pre-trained model and cache it for optimized performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('D:/Image_Classification/Flower_Image_Classification_For_12_ClassesH5.h5')  # Loading the pre-trained model
    return model  # Returning the loaded model

def preprocess_image(image):
    image = tf.image.resize(image, [150, 150])  # Resize the image
    image = tf.cast(image, tf.float32) / 255.0  # Rescale the pixel values to [0, 1]
    return image

def predict_class(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Adding an extra dimension to match the model's input requirements
    prediction = model.predict(image)  # Making predictions using the model
    return prediction

model = load_model()  # Loading the pre-trained model
st.title('Flower Classifier')  # Setting the title of the Streamlit app as 'Flower Classifier'

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])
# Creating a file uploader component for uploading images of type JPG or PNG

if file is None:
    st.text('Waiting for upload....')
    # Displaying a message if no file is uploaded yet

else:
    slot = st.empty()
    slot.text('Running inference....')
    # Displaying a message indicating the inference process is ongoing

    test_image = Image.open(file)
    # Opening the uploaded image using PIL's Image module

    st.image(test_image, caption="Input Image", width=400)
    # Displaying the uploaded image with a caption and width specified

    pred = predict_class(np.asarray(test_image), model)
    # Making predictions on the uploaded image using the loaded model

    class_names = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy','carnation','common_daisy','coreopsis','daffodil','dandelion','iris','magnolia','rose','sunflower','tulip','water_lily']
    # Defining class names for different flower types

    result = class_names[np.argmax(pred)]
    # Determining the predicted class by selecting the one with the highest probability

    output = 'The image is a ' + result
    # Generating the output message indicating the predicted flower class

    slot.text('Done')
    # Displaying a message indicating the completion of inference

    st.success(output)
    # Displaying the output message as a success notification in the Streamlit app
