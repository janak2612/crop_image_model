import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

# Streamlit app header
st.title("Image Classification App")

# Disable scientific notation for clarity in NumPy arrays
np.set_printoptions(suppress=True)

# Custom DepthwiseConv2D layer (if needed)
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the model
try:
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("Labels file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading labels: {e}")
    st.stop()

# Function to preprocess image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
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
    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        # Extract class name
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Display prediction results
        st.write(f"**Class:** {class_name}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
