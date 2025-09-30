import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Set page configuration for a more professional look
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

@st.cache_resource
def load_keras_model_and_labels():
    """
    Loads the Keras model and class labels from disk.
    The @st.cache_resource decorator ensures this function only runs once,
    speeding up the app on subsequent runs by caching the loaded model.
    """
    try:
        model = load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
        return model, class_names
    except FileNotFoundError:
        st.error("Error: `keras_Model.h5` or `labels.txt` not found.")
        st.error("Please make sure both files are in the same directory as `app.py`.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None


# Load the model and class names
model, class_names = load_keras_model_and_labels()

def classify_image(image_to_classify, model_to_use, labels):
    """
    Takes an image, a model, and class names, and returns the
    predicted class and confidence score.
    """
    if model_to_use is None or labels is None:
        return None, None

    # Create a data array with the correct shape for the Keras model.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize the image to 224x224 and crop from the center.
    size = (224, 224)
    image = ImageOps.fit(image_to_classify, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array.
    image_array = np.asarray(image)

    # Normalize the image data (pixel values from -1 to 1).
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the normalized image data into the array.
    data[0] = normalized_image_array

    # Run the prediction.
    prediction = model_to_use.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score


# --- Streamlit App Interface ---

st.title("Image Classification with Keras")
st.header("Upload an Image to Classify It")
st.write(
    "This app uses a pre-trained Keras model to predict the content of an image. "
    "Upload a JPG, JPEG, or PNG file and click the 'Classify' button."
)

# File uploader widget allows user to upload an image.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Open and display the uploaded image.
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Add a button to trigger classification.
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            # Make prediction.
            class_name, confidence_score = classify_image(image, model, class_names)
            
            # Display the results.
            st.success("Classification Complete!")
            st.metric(label="Prediction", value=f"{class_name[2:].strip()}")
            st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")

