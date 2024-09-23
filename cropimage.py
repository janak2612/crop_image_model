import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Streamlit app header
st.title("Image Classification App")

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Streamlit file uploader to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image(image)

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display prediction results
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
