import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np

st.set_page_config(page_title="Nhóm 10 Eval", page_icon="📝", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Nhóm 10 Eval</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Nhập text của bạn và nhận kết quả từ mô hình</p>", unsafe_allow_html=True)

def process(comments):
    with open('notebook/models/tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json_str = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json_str)
    sequences = tokenizer.texts_to_sequences(comments)
    pad = pad_sequences(sequences, padding='post', maxlen=30, truncating='post')
    model = load_model('notebook/models/btlpython3.h5')
    predictions = model.predict(pad)
    return np.argmax(predictions, axis=1)
@st.cache_resource
def load_model_tokenizer():
    model = load_model("notebook/models/btlpython3.h5")

    with open("notebook/models/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json_str = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json_str)
    return model, tokenizer



st.markdown("### Nhập văn bản để đánh giá")
user_input = st.text_area("Nhập vào đây...", height=150)

if st.button("Đánh giá"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập văn bản trước khi đánh giá!")
    else:
        sentiment = [
            "Negative",
            "Neutral",
            "Positive"
        ]
        model, tokenizer = load_model_tokenizer()
        st.markdown(f"<h3 style='color:#4B0082;'>Kết quả: {sentiment[int(process([user_input]))]}</h3>", unsafe_allow_html=True)

