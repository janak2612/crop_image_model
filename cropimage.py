import os
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, DepthwiseConv2D, Dense

# --- INITIAL SETUP ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
np.set_printoptions(suppress=True)

# --- CUSTOM LAYER FOR MODEL COMPATIBILITY ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:
            del kwargs["groups"]
        super().__init__(*args, **kwargs)

# --- MODEL AND LABEL LOADING ---
@st.cache_resource
def load_and_rebuild_model():
    """
    Loads the original Keras model and rebuilds it with a correct
    single-input structure. This version is more robust.
    """
    try:
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        base_model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)

        # --- ROBUST MODEL REBUILDING LOGIC ---
        # Find the core functional model and the classification head automatically.
        core_model_layer = None
        classification_layer = None

        # A Teachable Machine model is a Sequential model containing another Model and a Dense layer.
        for layer in base_model.layers:
            if isinstance(layer, Model): # Find the main functional model
                core_model_layer = layer
            elif isinstance(layer, Dense): # Find the final classification layer
                classification_layer = layer
        
        if core_model_layer is None or classification_layer is None:
            st.error("❌ **Model Structure Error:** Could not automatically identify the core and classification layers in the model. The model structure may be unexpected.")
            return None

        # 1. Create a new, single input tensor.
        new_input = Input(shape=(224, 224, 3), name="new_input")
        
        # 2. Pass the new input through the identified core model.
        core_output = core_model_layer(new_input)

        # 3. Pass the result through the identified classification layer.
        new_output = classification_layer(core_output)

        # 4. Create the final, corrected model.
        model = Model(inputs=new_input, outputs=new_output)
        
        return model
    except Exception as e:
        st.error(f"❌ **Model Loading Error:** Failed to load or rebuild the model. Please check the 'keras_model.h5' file. \n\n*Details: {e}*")
        return None

@st.cache_data
def load_labels():
    """Loads class labels from a text file."""
    try:
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        st.error("❌ **File Not Found:** The 'labels.txt' file is missing.")
        return None
    except Exception as e:
        st.error(f"❌ **Label Loading Error:** Could not read the labels file. \n\n*Details: {e}*")
        return None

# Load the model and labels
model = load_and_rebuild_model()
class_names = load_labels()

# --- IMAGE PREPROCESSING ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes and normalizes a PIL image for the model."""
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# --- STREAMLIT APP UI ---
st.title("🖼️ Image Classification App")
st.write("Upload an image, and the model will predict its class.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None and class_names is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.divider()
    
    with st.spinner("🔍 Analyzing image..."):
        try:
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = preprocess_image(image)
            
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            st.success("✅ Prediction Complete!")
            st.metric(label="**Predicted Class**", value=class_name)
            st.metric(label="**Confidence Score**", value=f"{confidence_score:.2%}")

        except Exception as e:
            st.error(f"❌ **Prediction Error:** An error occurred during the analysis. \n\n*Details: {e}*")

elif uploaded_file is None:
    st.info("Please upload an image file to begin.")
