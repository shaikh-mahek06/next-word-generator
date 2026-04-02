import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Next Word Generator", layout="centered")

# ------------------------------
# Custom Pastel Green UI
# ------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8f4);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Glass container */
.main-box {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 16px;
    border: 1px solid rgba(0, 80, 40, 0.15);
    box-shadow: 0 8px 25px rgba(0, 80, 40, 0.08);
}

/* Title */
h1 {
    text-align: center;
    color: #0b3d2e;
    font-weight: 600;
}

/* Input */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.7);
    border: 1px solid #b7d3c2;
    border-radius: 10px;
    padding: 12px;
    color: #0b3d2e;
}

/* Button */
.stButton > button {
    background-color: #0b3d2e;
    color: #ffffff;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    transition: 0.25s;
}

.stButton > button:hover {
    background-color: #145c43;
}

/* Result box */
.result-box {
    margin-top: 20px;
    padding: 16px;
    border-radius: 12px;
    background: rgba(232, 245, 233, 0.7);
    border: 1px solid #c8e6c9;
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    color: #0b3d2e;
}

/* Footer */
footer {
    text-align: center;
    color: #4f6f64;
    margin-top: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
        
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# Reverse word index
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# ------------------------------
# Prediction Function (10 words)
# ------------------------------
def predict_next_words(text, num_words=10):
    text = text.lower()
    
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([text])
        sequence = pad_sequences(sequence, maxlen=max_len-1, padding='pre')

        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)

        next_word = index_to_word.get(predicted_index, "")

        if next_word == "":
            break

        text += " " + next_word

    return text

# ------------------------------
# UI Layout
# ------------------------------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown("<h1>Next Word Generator</h1>", unsafe_allow_html=True)

user_input = st.text_input("", placeholder="Type your sentence...")

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Enter some text first.")
    else:
        generated_text = predict_next_words(user_input, num_words=10)
        
        st.markdown(f"""
        <div class="result-box">
            {generated_text}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<footer> Text Generator</footer>", unsafe_allow_html=True)