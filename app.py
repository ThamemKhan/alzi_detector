import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# Load the model
model = load_model("alzheimer_model.h5", compile=False)
model.make_predict_function()

# Label mapping
verbose_name = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented",
}

def predict_label(img):
    test_image = img.convert("L").resize((128, 128))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(-1, 128, 128, 1)
    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    return verbose_name[classes_x[0]]

# Streamlit UI
st.set_page_config(page_title="Alzheimer's Detection", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer's Disease Classification")
st.write("Upload an MRI image to classify the Alzheimer's disease stage.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        result = predict_label(img)
        st.success(f"Prediction: {result}")
        
        # Save uploaded image
        save_path = "static/uploads"
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, uploaded_file.name))
        st.write(f"Image saved to {save_path}/{uploaded_file.name}")
