import streamlit as st
import numpy as np
import pickle
import json
import re
import os
from huggingface_hub import hf_hub_download

# ── Page config ──
st.set_page_config(
    page_title="CV Job Classifier",
    page_icon="📄",
    layout="centered"
)

# ── Download model dari Hugging Face ──
REPO_ID = "ariif-rahmaan/cv-job-classifier-models"

@st.cache_resource
def load_models():
    st.info("Memuat model... (hanya sekali, mohon tunggu)")

    # Download semua file
    config_path    = hf_hub_download(REPO_ID, "config.json")
    le_path        = hf_hub_download(REPO_ID, "label_encoder.pkl")
    tok_path       = hf_hub_download(REPO_ID, "tokenizer_keras.pkl")
    lstm_path      = hf_hub_download(REPO_ID, "model_lstm.keras")
    gru_path       = hf_hub_download(REPO_ID, "model_gru.keras")
    bert_path      = hf_hub_download(REPO_ID, "bert_best.pt")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load label encoder
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    # Load tokenizer
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load LSTM & GRU
    import tensorflow as tf
    model_lstm = tf.keras.models.load_model(lstm_path)
    model_gru  = tf.keras.models.load_model(gru_path)

    # Load BERT
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model     = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=config['NUM_CLASSES'])
    bert_model.load_state_dict(
        torch.load(bert_path, map_location=torch.device('cpu')))
    bert_model.eval()

    return config, le, tokenizer, model_lstm, model_gru, bert_tokenizer, bert_model

# ── Preprocessing ──
def preprocess(text):
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ── Prediksi ──
def predict(text, config, le, tokenizer, model_lstm, model_gru, bert_tokenizer, bert_model):
    import torch
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    clean = preprocess(text)
    MAX_LEN      = config['MAX_LEN']
    MAX_LEN_BERT = config['MAX_LEN_BERT']
    results      = {}

    # LSTM
    seq  = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = np.argmax(model_lstm.predict(padded, verbose=0), axis=1)[0]
    results['LSTM'] = le.classes_[pred]

    # GRU
    pred = np.argmax(model_gru.predict(padded, verbose=0), axis=1)[0]
    results['GRU'] = le.classes_[pred]

    # BERT
    encoding = bert_tokenizer(
        clean, max_length=MAX_LEN_BERT,
        padding='max_length', truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = bert_model(**encoding)
    pred = output.logits.argmax(dim=1).item()
    results['BERT'] = le.classes_[pred]

    return results

# ── UI ──
st.title("📄 CV Job Classifier")
st.markdown("Sistem klasifikasi kategori pekerjaan berbasis CV menggunakan **LSTM**, **GRU**, dan **BERT**")
st.markdown("---")

# Load model
config, le, tokenizer, model_lstm, model_gru, bert_tokenizer, bert_model = load_models()
st.success("✓ Model berhasil dimuat!")

# Input
st.subheader("Input CV")
cv_text = st.text_area(
    "Paste teks CV di sini:",
    height=300,
    placeholder="Contoh: Experienced software developer with 5 years in Python, Django, REST APIs..."
)

model_choice = st.multiselect(
    "Pilih model:",
    ["LSTM", "GRU", "BERT"],
    default=["LSTM", "GRU", "BERT"]
)

if st.button("🔍 Klasifikasi", type="primary"):
    if not cv_text.strip():
        st.warning("Masukkan teks CV terlebih dahulu!")
    else:
        with st.spinner("Menganalisis CV..."):
            results = predict(
                cv_text, config, le, tokenizer,
                model_lstm, model_gru,
                bert_tokenizer, bert_model
            )

        st.markdown("---")
        st.subheader("Hasil Klasifikasi")

        cols = st.columns(len(model_choice))
        for i, model in enumerate(model_choice):
            with cols[i]:
                kategori = results[model].replace('_', ' ')
                st.markdown(f"**Model {model}**")
                st.success(kategori)
                
        # Majority vote
        votes = [results[m] for m in model_choice]
        final = max(set(votes), key=votes.count)
        st.markdown("---")
        st.success(f"**Kesimpulan:** CV ini paling cocok untuk posisi **{final.replace('_', ' ')}**")
