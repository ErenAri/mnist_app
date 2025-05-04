import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from datetime import datetime
import os
from PIL import Image

# Dil sözlüğü
translations = {
    "tr": {
        "title": "🖌️ El Yazısı Rakam Tanıma",
        "draw_prompt": "Bir rakam çizin (0-9) ve modelin ne tahmin ettiğini görün.",
        "predict_btn": "🎯 Tahmin Et",
        "clear_btn": "🗑️ Tuvali Temizle",
        "correct_question": "Bu tahmin doğru muydu?",
        "yes": "✅ Evet",
        "no": "❌ Hayır",
        "correct_label": "Doğru rakam kaçtı?",
        "thanks": "Teşekkürler, geri bildiriminiz kaydedildi!",
        "score": "🎯 Puanınız:",
        "streak": "🔥 Seri:",
        "matrix_title": "📊 Model Doğruluk Analizi (Confusion Matrix)",
        "matrix_caption": "Eğitim sonrası doğruluk analizi",
        "correct": "Harika! +1 puan kazandınız.",
        "nickname": "🧑 Adınızı girin (Nickname):",
        "leaderboard": "🏆 Skor Tablosu (İlk 5)"
    },
    "en": {
        "title": "🖌️ Handwritten Digit Recognition",
        "draw_prompt": "Draw a digit (0–9) and see what the model predicts.",
        "predict_btn": "🎯 Predict",
        "clear_btn": "🗑️ Clear Canvas",
        "correct_question": "Was this prediction correct?",
        "yes": "✅ Yes",
        "no": "❌ No",
        "correct_label": "What was the correct digit?",
        "thanks": "Thanks! Your feedback has been recorded.",
        "score": "🎯 Your Score:",
        "streak": "🔥 Streak:",
        "matrix_title": "📊 Model Accuracy Analysis (Confusion Matrix)",
        "matrix_caption": "Post-training accuracy overview",
        "correct": "Great! +1 point earned.",
        "nickname": "🧑 Enter your nickname:",
        "leaderboard": "🏆 Leaderboard (Top 5)"
    }
}

st.set_page_config(page_title="MNIST Tahmin Uygulaması")

# Dil seçimi
lang = st.selectbox("🌐 Dil / Language", ["tr", "en"], index=0)
st.session_state["lang"] = lang
_ = translations[lang]

# Nickname
nickname = st.text_input(_["nickname"], max_chars=20)
if "nickname" not in st.session_state and nickname:
    st.session_state.nickname = nickname

st.title(_["title"])
st.write(_["draw_prompt"])

# Modeli yükle
model = tf.keras.models.load_model("mnist_model.h5")

# Session state başlat
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0

# Temizle butonu
if st.button(_["clear_btn"]):
    st.session_state.clear_canvas = True
    st.rerun()

# Tuval bileşeni
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
    initial_drawing=None
)

# Tahmin işlemi
if st.button(_["predict_btn"]):
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
        st.subheader(f"🔢 {predicted_class} (%{confidence:.1f})")

        feedback = st.radio(_["correct_question"], [_["yes"], _["no"]])
        correct_label = ""

        if feedback == _["yes"]:
            st.success(_["correct"])
            st.session_state.score += 1
            st.session_state.streak += 1

            # skor güncelle
            if nickname:
                score_file = "scores.csv"
                if os.path.exists(score_file):
                    scores_df = pd.read_csv(score_file)
                else:
                    scores_df = pd.DataFrame(columns=["nickname", "score"])

                if nickname in scores_df["nickname"].values:
                    scores_df.loc[scores_df["nickname"] == nickname, "score"] += 1
                else:
                    scores_df = pd.concat([scores_df, pd.DataFrame([{"nickname": nickname, "score": 1}])], ignore_index=True)

                scores_df.to_csv(score_file, index=False)

        elif feedback == _["no"]:
            correct_label = st.text_input(_["correct_label"])
            if correct_label and correct_label.isdigit() and 0 <= int(correct_label) <= 9:
                feedback_data = {
                    "timestamp": datetime.now().isoformat(),
                    "wrong_prediction": int(predicted_class),
                    "correct_label": int(correct_label),
                    "confidence": float(confidence),
                    "image_array": img_resized.flatten().tolist()
                }
                csv_path = "feedback.csv"
                df = pd.DataFrame([feedback_data])
                if os.path.exists(csv_path):
                    df.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(csv_path, index=False)
                st.info(_["thanks"])
                st.session_state.streak = 0

        # Çizimi PNG olarak kaydet
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = "predictions"
        os.makedirs(folder, exist_ok=True)

        if feedback == _["yes"]:
            filename = f"{timestamp}_pred{predicted_class}_correct.png"
        else:
            label_part = f"label{correct_label}" if correct_label else "unknown"
            filename = f"{timestamp}_pred{predicted_class}_wrong_{label_part}.png"

        save_path = os.path.join(folder, filename)
        img_pil = Image.fromarray(img_resized.astype(np.uint8))
        img_pil.save(save_path)

        # Skor gösterimi
        st.markdown(f"{_['score']} {st.session_state.score}")
        st.markdown(f"{_['streak']} {st.session_state.streak}")

# Confusion Matrix
st.subheader(_["matrix_title"])
st.image("confusion_matrix.png", caption=_["matrix_caption"], use_container_width=True)

# Skor tablosu
st.markdown(f"### {_['leaderboard']}")
if os.path.exists("scores.csv"):
    scores_df = pd.read_csv("scores.csv")
    top_scores = scores_df.sort_values(by="score", ascending=False).head(5).reset_index(drop=True)
    st.table(top_scores)
else:
    st.info("Henüz skor tablosu oluşturulmadı.")
