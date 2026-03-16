import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load trained model
model = load_model("pneumonia_detector_vgg16.h5")

# Title and description
st.title("🩻 Pneumonia Detection")
st.write("Upload a chest X-ray to predict whether it shows signs of **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')  # convert to RGB
    st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 224, 224, 3)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Output
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"⚠️ Pneumonia Detected with confidence: {prediction:.2%}")
    else:
        st.success(f"✅ Normal Chest X-ray with confidence: {(1 - prediction):.2%}")
