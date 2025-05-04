
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(page_title="MNIST Tahmin Uygulaması")
st.title("El Yazısı Rakam Tanıma")

st.write("Bir rakam çizin (0-9) ve modelin ne tahmin ettiğini görün.")

# Modeli yükle
model = tf.keras.models.load_model("mnist_model.h5")

# Tuval temizleme durumu
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

if st.button("🗑️ Tuvali Temizle"):
    st.session_state.clear_canvas = True
    st.rerun()

canvas_result = st_canvas(
    fill_color="#000000",             # dolgu rengi (boşluklar)
    stroke_width=12,
    stroke_color="#FFFFFF",           # çizim rengi (kalem: beyaz)
    background_color="#000000",       # tuval arka planı: siyah
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
    initial_drawing=None
)


if st.button("🎯 Tahmin Et"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28))
        img_resized = 255 - img_resized
        img_normalized = img_resized / 255.0
        img_input = img_normalized.reshape(1, 28, 28, 1)
        prediction = model.predict(img_input)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100
        st.subheader(f"🔢 Tahmin: {predicted_class} (%{confidence:.1f} emin)")

        st.subheader("📊 Model Doğruluk Analizi (Confusion Matrix)")
        st.image("confusion_matrix.png", caption="Eğitim sonrası doğruluk analizi", use_container_width=True)
