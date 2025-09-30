
import os
# Force TensorFlow to use CPU only (suppress CUDA errors if GPU not available)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

# Streamlit app header
st.title("🖼️ Image Classification App")

# Disable scientific notation for clarity in NumPy arrays
np.set_printoptions(suppress=True)

# Custom DepthwiseConv2D layer (for compatibility with saved model)
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:   # remove unsupported arg
            del kwargs["groups"]
        super().__init__(*args, **kwargs)

# Load the model
try:
    custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
    model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load class labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("❌ Labels file not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading labels: {e}")
    st.stop()

# Preprocess image function
def preprocess_image(image):
    size = (224, 224)  # model input size
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    try:
        # Preprocess
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = preprocess_image(image)
        
        # Prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Results
        st.success("✅ Prediction Complete!")
        st.write(f"**Class:** {class_name}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


