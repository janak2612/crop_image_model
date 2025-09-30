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
def load_model_and_labels():
    """
    Loads the Keras model and class labels from disk.
    The @st.cache_resource decorator ensures this function only runs once,
    speeding up the app on subsequent runs.
    """
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

# Load the model and class names
model, class_names = load_model_and_labels()

def predict(image, model, class_names):
    """
    Takes an image, a model, and class names, and returns the
    predicted class and confidence score.
    """
    # Create the array of the right shape to feed into the Keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score


# --- Streamlit App Interface ---

st.title("Image Classification with Teachable Machine")
st.header("Upload an image to classify it!")
st.write(
    "This app uses a trained Keras model to predict the class of an image. "
    "Upload a JPG, JPEG, or PNG file and see the magic happen. ✨"
)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    
    # Add a button to trigger classification
    if st.button("Classify Image"):
        st.write("Classifying...")
        
        # Make prediction
        class_name, confidence_score = predict(image, model, class_names)
        
        # Display the prediction and confidence score
        st.success(f"**Prediction:** {class_name[2:].strip()}")
        st.info(f"**Confidence Score:** {confidence_score:.2%}")
