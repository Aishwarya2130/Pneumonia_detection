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
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)


    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    prediction = model.predict(img_array)[0][0]


    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"⚠️ Pneumonia Detected with confidence: {prediction:.2%}")


        st.warning("###  Health Advice:\n"
                   "**Causes of Pneumonia:**\n"
                   "- Bacteria (*e.g., Streptococcus pneumoniae*)\n"
                   "- Viruses (such as influenza, COVID-19)\n"
                   "- Fungi (especially in those with weakened immune systems)\n\n"
                   "**Prevention Tips:**\n"
                   "- ️ Get vaccinated regularly\n"
                   "-  Wash hands frequently\n"
                   "-  Avoid smoking\n"
                   "-  Eat healthy and stay hydrated\n"
                   "- ️ Exercise to boost immunity\n"
                   "-  Seek medical help if symptoms appear (fever, cough, chest pain)")
    else:
        st.success(f"✅ Normal Chest X-ray with confidence: {(1 - prediction):.2%}")


        st.info("###  Message:\nPerfect! You are fit and fine. \n\n"
                "Keep maintaining a healthy lifestyle by eating well, exercising regularly, and getting enough rest. 💪")
