import streamlit as st
from PIL import Image

from src.inference import predict

st.set_page_config(
    page_title="Happy vs Sad Classifier",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Happy vs 😢 Sad Image Classifier")

st.write(
    "Upload an image and the model will predict whether it is Happy or Sad."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded is not None:

    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            label, confidence = predict(image)

        st.success(f"Prediction: {label}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )