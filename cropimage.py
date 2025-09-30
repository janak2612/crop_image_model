import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Classifier",
    page_icon="🌾",
    layout="wide"
)

# --- MODEL AND LABELS LOADING ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model, caching it to improve performance."""
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ **Error Loading Model:** Could not load 'keras_model.h5'. Please ensure the file is in the correct directory and is not corrupted. \n\n*Details: {e}*")
        return None

@st.cache_data
def load_class_labels(labels_path):
    """Loads the class labels, caching them to improve performance."""
    try:
        with open(labels_path, "r") as f:
            class_names = f.readlines()
        return class_names
    except FileNotFoundError:
        st.error(f"❌ **Error Loading Labels:** 'labels.txt' not found. Please ensure the file is in the correct directory.")
        return None

# Load the model and labels from your files
model = load_keras_model("keras_model.h5")
class_names = load_class_labels("labels.txt")

# --- STREAMLIT APP UI ---
st.title("🌾 Crop Image Classifier")
st.write("Upload an image of jute, maize, rice, sugarcane, or wheat for classification.")

# Create the file uploader widget
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Ensure model and labels are loaded before proceeding
    if model is not None and class_names is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(image, caption="Image to be classified", use_column_width=False, width=300)

        # Add a button to trigger the prediction
        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing..."):
                # --- IMAGE PREPROCESSING ---
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                size = (224, 224)
                image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image_resized)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data[0] = normalized_image_array

                # --- PREDICTION ---
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # --- DISPLAY RESULTS IN COLUMNS ---
                st.divider()
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    st.success("✅ Prediction Complete!")
                    st.subheader("Results")
                    st.metric(label="**Predicted Crop**", value=class_name[2:].strip())
                    st.metric(label="**Confidence Score**", value=f"{confidence_score:.2%}")

else:
    st.info("Please upload an image file to get started.")

