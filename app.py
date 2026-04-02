import streamlit as st
import numpy as np
import pickle
import json
import re
import os
from huggingface_hub import hf_hub_download

# ── Page config ──
st.set_page_config(
    page_title="CV Job Classifier — HR Mode",
    page_icon="📋",
    layout="wide"
)

REPO_ID = "ariif-rahmaan/cv-job-classifier-models"

KATEGORI_LIST = [
    "Database_Administrator",
    "Network_Administrator",
    "Project_manager",
    "Security_Analyst",
    "Software_Developer",
    "Systems_Administrator"
]

@st.cache_resource
def load_models():
    with st.spinner("Memuat model... (hanya sekali, mohon tunggu)"):
        config_path = hf_hub_download(REPO_ID, "config.json")
        le_path     = hf_hub_download(REPO_ID, "label_encoder.pkl")
        tok_path    = hf_hub_download(REPO_ID, "tokenizer_keras.pkl")
        lstm_path   = hf_hub_download(REPO_ID, "model_lstm.keras")
        gru_path    = hf_hub_download(REPO_ID, "model_gru.keras")
        bert_path   = hf_hub_download(REPO_ID, "bert_best.pt")

        with open(config_path) as f:
            config = json.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)

        import tensorflow as tf
        model_lstm = tf.keras.models.load_model(lstm_path)
        model_gru  = tf.keras.models.load_model(gru_path)

        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model     = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=config['NUM_CLASSES'])
        bert_model.load_state_dict(
            torch.load(bert_path, map_location=torch.device('cpu')))
        bert_model.eval()

    return config, le, tokenizer, model_lstm, model_gru, bert_tokenizer, bert_model

# ── Translate ──
from deep_translator import GoogleTranslator
from langdetect import detect

def translate_if_needed(text):
    try:
        lang = detect(text)
        if lang != 'en':
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated = ""
            for chunk in chunks:
                translated += GoogleTranslator(source='auto', target='en').translate(chunk)
            return translated, lang
        return text, 'en'
    except Exception:
        return text, 'en'

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

# ── Prediksi dengan confidence ──
def predict_with_confidence(text, config, le, tokenizer,
                             model_lstm, model_gru,
                             bert_tokenizer, bert_model):
    import torch
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    clean    = preprocess(text)
    MAX_LEN  = config['MAX_LEN']
    MAX_LEN_BERT = config['MAX_LEN_BERT']
    results  = {}

    # LSTM
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    proba_lstm = model_lstm.predict(padded, verbose=0)[0]
    pred_lstm  = np.argmax(proba_lstm)
    results['LSTM'] = {
        'label'      : le.classes_[pred_lstm],
        'confidence' : float(proba_lstm[pred_lstm]),
        'all_proba'  : {le.classes_[i]: float(p)
                        for i, p in enumerate(proba_lstm)}
    }

    # GRU
    proba_gru = model_gru.predict(padded, verbose=0)[0]
    pred_gru  = np.argmax(proba_gru)
    results['GRU'] = {
        'label'      : le.classes_[pred_gru],
        'confidence' : float(proba_gru[pred_gru]),
        'all_proba'  : {le.classes_[i]: float(p)
                        for i, p in enumerate(proba_gru)}
    }

    # BERT
    encoding = bert_tokenizer(
        clean, max_length=MAX_LEN_BERT,
        padding='max_length', truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = bert_model(**encoding)
    proba_bert = torch.softmax(output.logits, dim=1)[0].numpy()
    pred_bert  = np.argmax(proba_bert)
    results['BERT'] = {
        'label'      : le.classes_[pred_bert],
        'confidence' : float(proba_bert[pred_bert]),
        'all_proba'  : {le.classes_[i]: float(p)
                        for i, p in enumerate(proba_bert)}
    }

    # Voting
    votes      = [results[m]['label'] for m in ['LSTM', 'GRU', 'BERT']]
    vote_count = {k: votes.count(k) for k in set(votes)}
    final      = max(vote_count, key=vote_count.get)

    # Kalau seri → percaya BERT
    if list(vote_count.values()).count(max(vote_count.values())) > 1:
        final = results['BERT']['label']

    # Confidence score final = rata-rata confidence untuk kategori final
    conf_final = np.mean([
        results[m]['all_proba'].get(final, 0)
        for m in ['LSTM', 'GRU', 'BERT']
    ])

    # Flag out-of-domain kalau confidence rendah
    out_of_domain = conf_final < 0.4

    return results, final, conf_final, vote_count, out_of_domain

# ── Extract teks dari PDF ──
def extract_pdf(file):
    import pdfplumber
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ══════════════════════════════════════════
# UI UTAMA
# ══════════════════════════════════════════

st.title("📋 Sistem Seleksi CV Otomatis")
st.markdown("Sistem klasifikasi CV berbasis **LSTM**, **GRU**, dan **BERT** untuk membantu HR menyaring kandidat.")
st.markdown("---")

# Load model
config, le, tokenizer, model_lstm, model_gru, bert_tokenizer, bert_model = load_models()
st.success("✓ Model berhasil dimuat!")

# ── Step 1: HR pilih posisi ──
st.header("Step 1 — Pilih Posisi yang Dicari")
posisi_display = {k.replace('_', ' '): k for k in KATEGORI_LIST}
posisi_pilihan = st.selectbox(
    "Posisi:",
    list(posisi_display.keys())
)
posisi_target = posisi_display[posisi_pilihan]
st.info(f"HR mencari kandidat untuk posisi: **{posisi_pilihan}**")

st.markdown("---")

# ── Step 2: Upload CV ──
st.header("Step 2 — Upload CV Kandidat")
uploaded_files = st.file_uploader(
    "Upload satu atau beberapa file CV (PDF):",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} CV berhasil diupload!")

st.markdown("---")

# ── Step 3: Analisis ──
st.header("Step 3 — Analisis & Ranking")

if st.button("🔍 Analisis CV", type="primary"):
    if not uploaded_files:
        st.warning("Upload minimal 1 CV terlebih dahulu!")
    else:
        all_results = []

        progress = st.progress(0)
        status   = st.empty()

        for i, file in enumerate(uploaded_files):
            status.text(f"Menganalisis {file.name}...")

            # Extract teks
            raw_text = extract_pdf(file)

            # Translate
            translated_text, lang = translate_if_needed(raw_text)

            # Prediksi
            results, final, conf_final, vote_count, out_of_domain = predict_with_confidence(
                translated_text, config, le, tokenizer,
                model_lstm, model_gru, bert_tokenizer, bert_model
            )

            # Cocok dengan posisi target?
            is_match = final == posisi_target

            all_results.append({
                'nama_file'    : file.name,
                'bahasa'       : lang,
                'hasil_lstm'   : results['LSTM']['label'].replace('_', ' '),
                'conf_lstm'    : results['LSTM']['confidence'],
                'hasil_gru'    : results['GRU']['label'].replace('_', ' '),
                'conf_gru'     : results['GRU']['confidence'],
                'hasil_bert'   : results['BERT']['label'].replace('_', ' '),
                'conf_bert'    : results['BERT']['confidence'],
                'final'        : final.replace('_', ' '),
                'confidence'   : conf_final,
                'is_match'     : is_match,
                'out_of_domain': out_of_domain,
                'vote_count'   : vote_count
            })

            progress.progress((i + 1) / len(uploaded_files))

        status.text("✓ Analisis selesai!")

        # ── Tampilkan hasil ──
        st.markdown("---")
        st.subheader(f"Hasil Analisis — Posisi: {posisi_pilihan}")

        # Pisahkan yang cocok dan tidak
        cocok     = [r for r in all_results if r['is_match'] and not r['out_of_domain']]
        tidak_cocok = [r for r in all_results if not r['is_match'] or r['out_of_domain']]

        # Urutkan berdasarkan confidence
        cocok     = sorted(cocok, key=lambda x: x['confidence'], reverse=True)
        tidak_cocok = sorted(tidak_cocok, key=lambda x: x['confidence'], reverse=True)

        # ── CV yang cocok ──
        st.markdown(f"### ✅ CV yang Cocok ({len(cocok)} kandidat)")
        if cocok:
            for rank, r in enumerate(cocok, 1):
                with st.expander(
                    f"#{rank} — {r['nama_file']} "
                    f"| Confidence: {r['confidence']*100:.1f}%"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("LSTM", r['hasil_lstm'],
                                  f"{r['conf_lstm']*100:.1f}%")
                    with col2:
                        st.metric("GRU", r['hasil_gru'],
                                  f"{r['conf_gru']*100:.1f}%")
                    with col3:
                        st.metric("BERT", r['hasil_bert'],
                                  f"{r['conf_bert']*100:.1f}%")
                    with col4:
                        st.metric("Kesimpulan", r['final'],
                                  f"{r['confidence']*100:.1f}%")

                    if r['bahasa'] != 'en':
                        st.caption(f"CV terdeteksi bahasa: {r['bahasa']} — otomatis diterjemahkan")
        else:
            st.warning("Tidak ada CV yang cocok dengan posisi ini.")

        # ── CV yang tidak cocok ──
        st.markdown(f"### ❌ CV yang Tidak Cocok ({len(tidak_cocok)} kandidat)")
        if tidak_cocok:
            for r in tidak_cocok:
                with st.expander(f"— {r['nama_file']} | Prediksi: {r['final']}"):
                    if r['out_of_domain']:
                        st.error(
                            "⚠️ CV ini kemungkinan berada di luar domain IT — "
                            "confidence score sangat rendah."
                        )
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LSTM", r['hasil_lstm'],
                                  f"{r['conf_lstm']*100:.1f}%")
                    with col2:
                        st.metric("GRU", r['hasil_gru'],
                                  f"{r['conf_gru']*100:.1f}%")
                    with col3:
                        st.metric("BERT", r['hasil_bert'],
                                  f"{r['conf_bert']*100:.1f}%")

        # ── Ringkasan ──
        st.markdown("---")
        st.subheader("Ringkasan")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total CV", len(all_results))
        with c2:
            st.metric("CV Cocok", len(cocok))
        with c3:
            st.metric("CV Tidak Cocok", len(tidak_cocok))
