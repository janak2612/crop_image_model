
import os
import warnings

# Suppress TensorFlow GPU/CUDA warnings (force CPU mode)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress Streamlit ScriptRunContext warning
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import DepthwiseConv2D, Input
from PIL import Image, ImageOps

# App title
st.title("🖼️ Image Classification App")

# Disable scientific notation in NumPy
np.set_printoptions(suppress=True)

# Custom DepthwiseConv2D for compatibility
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:  # remove unsupported arg
            del kwargs["groups"]
        super().__init__(*args, **kwargs)

# Load and fix model
try:
    custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
    base_model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)

    # If model has multiple inputs (Teachable Machine issue), keep only first input
    if isinstance(base_model.inputs, list) and len(base_model.inputs) > 1:
        model = Model(inputs=base_model.inputs[0], outputs=base_model.outputs)
    else:
        model = base_model

    # Optional: rewrap with proper Input layer to avoid Dense input_shape warnings
    if not hasattr(model, "_is_compiled"):  # crude check
        inputs = Input(shape=(224, 224, 3))
        outputs = model(inputs)
        model = Model(inputs=inputs, outputs=outputs)

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("❌ Labels file not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading labels: {e}")
    st.stop()

# Image preprocessing function
def preprocess_image(image):
    size = (224, 224)  # expected model input size
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("🔍 Analyzing image..."):
        try:
            # Preprocess
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = preprocess_image(image)
            
            # Prediction
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            # Show results
            st.success("✅ Prediction complete!")
            st.write(f"**Class:** {class_name}")
            st.write(f"**Confidence Score:** {confidence_score:.2f}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
