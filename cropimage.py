import os
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, DepthwiseConv2D, Dense

# --- INITIAL SETUP ---
# Suppress TensorFlow GPU/CUDA warnings by forcing CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress a specific Streamlit warning about ScriptRunContext
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
# Disable scientific notation for NumPy arrays for cleaner output
np.set_printoptions(suppress=True)

# --- CUSTOM LAYER FOR MODEL COMPATIBILITY ---
# This custom class handles older Keras models saved with an unsupported 'groups' argument.
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:
            del kwargs["groups"]
        super().__init__(*args, **kwargs)

# --- MODEL AND LABEL LOADING (CACHED FOR PERFORMANCE) ---
@st.cache_resource
def load_and_rebuild_model():
    """
    Loads the original Keras model and performs a deep rebuild to fix severe
    structural issues from some Teachable Machine model versions.
    """
    try:
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        # Load the base model, which contains the flawed internal structure
        base_model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)

        # --- Deep Model Rebuilding Logic ---
        # Find the main functional model and the final classification layer.
        core_model_layer = None
        classification_layer = None
        for layer in base_model.layers:
            if isinstance(layer, Model):
                core_model_layer = layer
            elif isinstance(layer, Dense):
                classification_layer = layer

        if core_model_layer is None or classification_layer is None:
            st.error("❌ **Model Structure Error:** Could not automatically identify the core and classification layers. The model structure may be unexpected.")
            return None

        # 1. Create a new, single input tensor. This is our clean starting point.
        new_input = Input(shape=(224, 224, 3), name="new_input")

        # 2. Rebuild the model from the inside out.
        #    Take the layers from *inside* the flawed functional model and connect them manually.
        #    We start with our new input and skip the original model's broken input layer (`core_model_layer.layers[1:]`).
        x = new_input
        for layer in core_model_layer.layers[1:]:
            x = layer(x)

        # 3. Connect the output of our rebuilt core to the original classification layer.
        new_output = classification_layer(x)

        # 4. Create the final, fully corrected model.
        model = Model(inputs=new_input, outputs=new_output)
        
        return model
        
    except FileNotFoundError:
        st.error("❌ **Model File Not Found:** Make sure 'keras_model.h5' is in the same folder as this script.")
        return None
    except Exception as e:
        # This will catch the "truncated file" error if the file is corrupted.
        st.error(f"❌ **Model Loading Error:** Failed to load or rebuild the model. The file may be corrupted. Please re-download it. \n\n*Details: {e}*")
        return None

@st.cache_data
def load_labels():
    """Loads class labels from a text file."""
    try:
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        st.error("❌ **Labels File Not Found:** Make sure 'labels.txt' is in the same folder as this script.")
        return None
    except Exception as e:
        st.error(f"❌ **Label Loading Error:** Could not read the labels file. \n\n*Details: {e}*")
        return None

# Load the model and labels when the app starts
model = load_and_rebuild_model()
class_names = load_labels()

# --- IMAGE PREPROCESSING ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes and normalizes a PIL image for the model."""
    size = (224, 224)
    # Resize and crop the image to fit the target size without distortion
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    # Normalize the pixel values to the [-1, 1] range, as the model expects
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# --- STREAMLIT APP UI ---
st.title("🖼️ Image Classification App")
st.write("Upload an image, and the model will predict its class.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

# Only proceed if the model and labels loaded correctly and a file was uploaded
if uploaded_file is not None and model is not None and class_names is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.divider()
    
    # Show a spinner while processing the image
    with st.spinner("🔍 Analyzing image..."):
        try:
            # Create a batch of 1 for the model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = preprocess_image(image)
            
            # Make the prediction
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            # Display the results
            st.success("✅ Prediction Complete!")
            st.metric(label="**Predicted Class**", value=class_name)
            st.metric(label="**Confidence Score**", value=f"{confidence_score:.2%}")

        except Exception as e:
            st.error(f"❌ **Prediction Error:** An error occurred during the analysis. \n\n*Details: {e}*")

elif uploaded_file is None:
    st.info("Please upload an image file to begin.")

