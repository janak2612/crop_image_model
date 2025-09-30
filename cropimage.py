import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# --- MODEL AND LABELS LOADING ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model, caching it to improve performance."""
    try:
        # Load the model. The 'compile=False' is important for models from Teachable Machine.
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ **Error Loading Model:** Could not load 'keras_Model.h5'. Please ensure the file is in the correct directory and is not corrupted. \n\n*Details: {e}*")
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
model = load_keras_model("keras_Model.h5")
class_names = load_class_labels("labels.txt")

# --- STREAMLIT APP UI ---
st.title("🖼️ Teachable Machine Image Classifier")
st.write("Upload an image and the model will predict its class.")

# Create the file uploader widget
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Ensure model and labels are loaded before proceeding
    if model is not None and class_names is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.divider()

        # Add a button to trigger the prediction
        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing..."):
                # --- IMAGE PREPROCESSING ---
                # Create a data array with the correct shape for the model
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

                # Resize the image to 224x224 and crop from the center
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

                # Turn the image into a numpy array
                image_array = np.asarray(image)

                # Normalize the image (scale pixel values to -1 to 1)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

                # Load the preprocessed image into the data array
                data[0] = normalized_image_array

                # --- PREDICTION ---
                # Predict the model
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # --- DISPLAY RESULTS ---
                st.success("✅ Prediction Complete!")
                # Display the class name, stripping the leading number and space (e.g., "0 Cat" -> "Cat")
                st.metric(label="**Predicted Class**", value=class_name[2:].strip())
                # Display the confidence score as a percentage
                st.metric(label="**Confidence Score**", value=f"{confidence_score:.2%}")
else:
    st.info("Please upload an image file to get started.")

