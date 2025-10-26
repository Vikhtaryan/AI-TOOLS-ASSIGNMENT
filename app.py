# app.py - Streamlit MNIST Digit Classifier Web App

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Title of the web app
st.title("MNIST Handwritten Digit Classifier")

# Load the trained CNN model (make sure mnist_cnn_model.h5 is in the same directory)
@st.cache_resource  # Cache model loading to optimize performance
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# Function to preprocess uploaded image for model prediction
def preprocess_image(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Invert image colors (white background, black digit)
    image = ImageOps.invert(image)
    
    # Resize to 28x28 pixels (MNIST image size)
    image = image.resize((28, 28))
    
    # Normalize pixel values (0 to 1)
    img_array = np.array(image) / 255.0
    
    # Reshape to (1, 28, 28, 1) for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# File uploader widget
uploaded_file = st.file_uploader("Upload an image file (PNG, JPG, JPEG) of a handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Show the uploaded image on the page
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for prediction
    processed_img = preprocess_image(image)
    
    # Predict the digit
    prediction = model.predict(processed_img)
    
    # Get the predicted class (digit 0-9)
    predicted_digit = np.argmax(prediction)
    
    # Show the prediction result
    st.success(f"Predicted Digit: {predicted_digit}")

else:
    st.info("Please upload an image to classify")
