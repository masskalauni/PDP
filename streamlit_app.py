import os
import json
import gdown
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Streamlit app

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)
# st.title("Load ML Model from Google Drive")
st.set_page_config(page_title="Plant Disease Prediction", page_icon="🪴", layout="wide")
# Google Drive file ID
# file_id = "1-2grAdMWW7G3HuP-nV9w-q1FOlU2L0g2"
#
# # Construct the Google Drive download URL
# model_url = f"https://drive.google.com/uc?id={file_id}"

# Model file path
# working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")

# Function to download the model from Google Drive
# @st.cache_resource
# def download_model(url, output):
#     gdown.download(url, output, quiet=False)
#     return output

# Download the model if it doesn't exist
# if not os.path.exists(model_path):
#     st.write("Downloading model...")
#     download_model(model_url, model_path)

# Load the model
# model = tf.keras.models.load_model(model_path)

# Display a message
# st.write("Model loaded successfully!")

# Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    img = image.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence_score = np.max(predictions)
    return predicted_class_name, confidence_score


# Streamlit App
# Set page configuration
# st.set_page_config(page_title="Plant Disease Prediction", page_icon="🪴", layout="wide")

# Sidebar
st.sidebar.title("🪴 Plant Disease Prediction")
st.sidebar.info("Upload an image of a plant leaf to predict its health status.")
st.sidebar.markdown("## How to Use")
st.sidebar.write(
    "1. Upload a clear image of a plant leaf.\n2. Click on the 'Predict' button.\n3. Wait for the model to analyze the image and display the result.")

# Main title and description
st.title('🪴 Plant Disease Prediction Model')
st.write("This model helps in predicting the health status of a plant leaf. Upload an image to get started.")

# File uploader
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# If image is uploaded
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)

        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded Image", use_column_width=False)

        # Image processing and prediction button
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Resized Image")
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            st.subheader("Prediction")
            if st.button('Predict'):
                with st.spinner('Predicting...'):
                    # Preprocess the uploaded image and predict the class
                    prediction, confidence = predict_image_class(model, image, class_indices)
                    st.success(f'Prediction: {str(prediction)}')
                    st.info(f'Confidence Score: {confidence:.2f}')
                    st.session_state.feedback_given = True

        if st.session_state.feedback_given:
            feedback = st.text_input('Was this prediction correct? (Yes/No)')
            if feedback.lower() in ['yes', 'no']:
                st.write("Thank you for your feedback!")
                st.session_state.feedback_given = False

    except Exception as e:
        st.error("First Submit the image ..")

# Additional information
st.markdown("---")
st.header("About the Model")
st.write("""
This model uses advanced machine learning techniques to analyze the image of a plant leaf and predict its health status. 
It can detect various diseases and provide insights on how to care for the plant.
""")

st.header("How to Use")
st.write("""
1. Upload a clear image of a plant leaf.
2. Click on the 'Predict' button.
3. Wait for the model to analyze the image and display the result.
""")

# Footer with About Us, Contact, and Copyright
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    text-align: center;
    padding: 10px;
}
.footer a {
    color: white;
    text-decoration: none;
    margin: 0 10px;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>&copy; 2024 Plant Disease Prediction Model | <a href="">About Us</a> | <a href="">Contact</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
