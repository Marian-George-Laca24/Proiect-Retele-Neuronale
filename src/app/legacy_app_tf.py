import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =========================
# CONFIGURARE GENERALĂ
# =========================

MODEL_PATH = "models/trained_model.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Bubble",
    "Crack",
    "Inclusion",
    "OK",
    "Scratch"
]

# Oprire mesaje TensorFlow inutile
tf.get_logger().setLevel("ERROR")

st.set_page_config(
    page_title="Detectare Defecte Sticlă",
    page_icon="🔍",
    layout="wide"
)

# =========================
# UI – HEADER
# =========================

st.title("Sistem de Detectare Defecte pe Sticlă")
st.markdown(
    """
    Aplicația utilizează o **rețea neuronală antrenată** pentru a detecta
    defecte vizuale pe suprafețe din sticlă industrială.
    """
)

st.divider()

# =========================
# ÎNCĂRCARE MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# UPLOAD IMAGINE
# =========================

uploaded_file = st.file_uploader(
    "Încărcați o imagine (JPG / PNG):",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESARE + INFERENȚĂ
# =========================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Imagine încărcată",
        width=400   # poți pune 350–450, 400 e echilibrat
    )

    # --- Preprocesare ---
    img = image.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # --- Predicție ---
    predictions = model.predict(img)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    predicted_class = CLASS_NAMES[class_index]

    st.divider()

    # =========================
    # AFIȘARE REZULTATE
    # =========================

    if predicted_class == "OK":
        st.success(f"✅ Sticlă fără defect (OK)")
    else:
        st.error(f"⚠️ Defect detectat: **{predicted_class}**")

    st.markdown(
        f"### 📊 Confidence: **{confidence * 100:.2f}%**"
    )

    # Bară probabilități pe clase
    st.subheader("Distribuția probabilităților pe clase")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}")
        st.progress(float(predictions[i]))

else:
    st.info("⬆Încărcați o imagine pentru a începe inferența.")
