import os
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow.keras.models
from tensorflow.keras.layers import DepthwiseConv2D

# --- INITIAL SETUP ---
# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message=".*GraphDef.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

# --- CUSTOM LAYER FOR MODEL COMPATIBILITY ---
# Teachable Machine models sometimes use a parameter that needs to be handled.
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:
            del kwargs["groups"]
        super().__init__(*args, **kwargs)

# --- CORE FUNCTIONS (CACHED FOR EFFICIENCY) ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    try:
        # Load the model with the custom layer to handle compatibility issues.
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        model = tensorflow.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ **Error Loading Model:** Failed to load 'keras_model.h5'. Please ensure the file is in the correct directory and is not corrupted. \n\n*Details: {e}*")
        return None

@st.cache_data
def load_labels(labels_path):
    """Loads the class labels from a text file."""
    try:
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        st.error(f"❌ **Error Loading Labels:** 'labels.txt' not found. Please ensure the file is in the correct directory.")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepares the uploaded image to be in the format the model expects.
    - Resizes to 224x224 pixels
    - Normalizes pixel values to be between -1 and 1
    """
    # Create a NumPy array to hold the image data
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Resize and crop the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    
    # Convert image to a NumPy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the normalized image data into the array
    data[0] = normalized_image_array
    
    return data

# --- STREAMLIT APP UI ---
st.title("🖼️ Image Classification App")
st.write("Upload an image and this app will predict what it is, based on the trained model.")

# Load the model and labels
model = load_keras_model("keras_model.h5")
class_names = load_labels("labels.txt")

# Create the file uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Check if model and labels are loaded successfully before proceeding
    if model is not None and class_names is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.divider()
        
        # Add a button to trigger the prediction
        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing..."):
                # Preprocess the image and make a prediction
                processed_data = preprocess_image(image)
                prediction = model.predict(processed_data)
                
                # Get the top prediction
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                
                # Display the results
                st.success("✅ Prediction Complete!")
                st.metric(label="**Predicted Class**", value=class_name.split(' ', 1)[1]) # Show class name after number
                st.metric(label="**Confidence Score**", value=f"{confidence_score:.2%}")
else:
    st.info("Please upload an image file to get started.")

