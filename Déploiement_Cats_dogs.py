import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Classification des chats et chiens", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_model.keras")

model = load_model()
col1, col2 = st.columns([1, 5])

with col1:
    st.image(
        "animaux-domestiques.png",
        width=100
    )

with col2:
    st.title("Classification des chats et chiens")
    st.subheader("Projet Deep Learning de la classification d'images")

st.write("Chargez une image de chien ou de chat et le modèle prédit la classe.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    

    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    proba = model.predict(img_array)[0][0]
    
    
    if proba > 0.5:
        label = "Chien"
        image_label = st.image(
            "labrador.png",
            width=100
        )
        confidence = proba
    else:
        label = "Chat"
        image_label = st.image(
            "smileys.png",
            width=100
        )
        confidence = 1 - proba

    st.subheader(f"Prédiction : **{label}**")
    st.write(f"Confiance : {confidence*100:.2f} %")
    st.image(image, caption="Image chargée", width=300)
