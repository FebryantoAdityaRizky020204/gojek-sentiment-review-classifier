import streamlit as st
import base64
import time

import pandas as pd
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
import json

import numpy as np
import string
import os
import pickle
import time
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from scipy.sparse import csr_matrix, hstack
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from sklearn.utils import class_weight
from keras.utils import to_categorical # type: ignore
from sklearn.utils.class_weight import compute_class_weight

from gensim.models import Word2Vec # type: ignore
from gensim.models import FastText # type: ignore
from keras.models import load_model # type: ignore

from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # type: ignore
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize # type: ignore
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # type: ignore
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk # type: ignore
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="HOME - Gojek Statement Review Classifier",
    page_icon="🏠",
    layout="wide"
)


# ==================================================================================

SVM_MODEL_PATH = "model/svm_tfidf_model.pkl"
SVM_VECTORIZER_PATH = "model/svm_tfidf_vectorizer.pkl"

NB_MODEL_PATH = "model/naive_bayes_bow_model.pkl"
NB_VECTORIZER_PATH = "model/naive_bayes_bow_vectorizer.pkl"


bilstm_label_encoder_path = "model/bilstm_label_encoder.pkl"
bilstm_w2v_path = "model/bilstm_w2v_model.keras"
bilstm_model_path = "model/bilstm_model.keras"

bigru_label_encoder_path = "model/gru_label_encoder.pkl"
bigru_fasttext_path = "model/gru_fasttext_model.keras"
bigru_model_path = "model/gru_model.keras"



@st.cache_resource
def load_svm_model_and_vectorizer():
    """Load the SVM model and vectorizer from disk."""
    with open(SVM_MODEL_PATH, 'rb') as model_file:
        svm_model = pickle.load(model_file)
    with open(SVM_VECTORIZER_PATH, 'rb') as vec_file:
        svm_vectorizer = pickle.load(vec_file)
    return svm_model, svm_vectorizer

svm_model, svm_vectorizer = load_svm_model_and_vectorizer()

@st.cache_resource
def load_naive_bayes_model_and_vectorizer():
    """Load the Naive Bayes model and vectorizer from disk."""
    with open(NB_MODEL_PATH, 'rb') as model_file:
        nb_model = pickle.load(model_file)
    with open(NB_VECTORIZER_PATH, 'rb') as vec_file:
        nb_vectorizer = pickle.load(vec_file)
    return nb_model, nb_vectorizer

nb_model, nb_vectorizer = load_naive_bayes_model_and_vectorizer()

@st.cache_resource
def load_bilstm_model_and_tokenizer():
    # Load model dan encoder
    bilstm_model = load_model(bilstm_model_path)
    w2v_model = Word2Vec.load(bilstm_w2v_path)

    with open(bilstm_label_encoder_path, "rb") as f:
        bilstm_label_encoder = pickle.load(f)
    return bilstm_model, w2v_model, bilstm_label_encoder

bilstm_model, w2v_model, bilstm_label_encoder = load_bilstm_model_and_tokenizer()

@st.cache_resource
def load_bigru_model_and_tokenizer():
    with open(bigru_label_encoder_path, "rb") as f:
        bigru_label_encoder = pickle.load(f)

    bigru_model = load_model(bigru_model_path)
    bigru_fasttext = FastText.load(bigru_fasttext_path)
    return bigru_model, bigru_fasttext, bigru_label_encoder

bigru_model, bigru_fasttext, bigru_label_encoder = load_bigru_model_and_tokenizer()

# === Preprocessing Function ===
stopwords_factory = StopWordRemoverFactory()
stopwords = set(stopwords_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(vectors, axis=0)

# ==================================================================================

with st.sidebar:
    selected = option_menu(
        menu_title="MODELS",
        options=["Traditional Machine Learning Model","Deep Learning Model"]
    )

    with st.container(border=True):
        st.markdown("### Kelompok 6")

        namaKelompok = pd.DataFrame({
            "Nama" : ["Yunisa Nur Safa", "Willy Azrieel", "Aditya Rizky Febryanto", "Novita Maria", "Milky Gratia Br Sitorus", "Melda Nia Yuliani", "Dectrixie Theodore Mayumi S."],
            "NIM" : ["223020503078", "223020503101", "223020503108", "223020503109", "223020503116", "223020503119", "223020503140"]
        })

        st.dataframe(
            namaKelompok,
            column_config={
                "AnggotaKelompok": st.column_config.ListColumn(
                    "Anggota Kelompok",
                    width="medium",
                ),
            },
            hide_index=True,
        )


st.title("Gojek Statement Review Classifier")
st.markdown("This application is designed to classify review statements into positive, neutral, or negative sentiments using traditional machine learning and deep learning models, built on Gojek review data")

with st.container(height=max(100, 320)):
    first_review = st.text_area(
        "Masukkan Review 👇",
        placeholder="Masukkan review anda di sini...",
        height=100
    )

    second_review = st.selectbox(
        "Atau Gunakan Contoh Ulasan Sentimen 👇",
        (
            "Pengemudi datang tepat waktu dan sangat sopan",
            "Pelayanan cukup baik, namun bisa lebih ditingkatkan",
            "Aplikasi sering mengalami gangguan saat digunakan",  
            "Pengemudinya ramah dan kendaraan bersih",            
            "Harga tidak sesuai dengan jarak tempuh",             
            "Proses pemesanan berjalan lancar",                   
            "Waktu tunggu terlalu lama",                          
            "Tidak ada masalah berarti saat menggunakan aplikasi",
        )
    )


    left_button, right_button = st.columns(2)
    left_button.button("Analisis Sentimen", key="submit_review", type="primary", use_container_width=True)
    right_button.button("Analisis Sentimen Contoh", key="submit_review_contoh", type="primary", use_container_width=True)


st.markdown("---")

text_negative = "<p style='font-weight:bold;font-size:20px'>&#9593; PREDIKSI SENTIMEN: <span style='color:red;'>NEGATIVE</span> &#9595;</p>"
text_netral = "<p style='font-weight:bold;font-size:20px'>&#9599; PREDIKSI SENTIMEN: <span style='color:grey;'>NETRAL</span> &#9599;</p>"
text_positive = "<p style='font-weight:bold;font-size:20px'>&#9595; PREDIKSI SENTIMEN: <span style='color:green;'>POSITIVE</span> &#9593;</p>"

data_dummy_negative = {"positive": 30,"neutral": 20,"negative": 50}
data_dummy_netral = {"positive": 30,"neutral": 50,"negative": 20}
data_dummy_positive = {"positive": 70,"neutral": 20,"negative": 10}


if selected == "Traditional Machine Learning Model":
    st.markdown("### Analisis Sentimen Menggunakan Traditional Machine Learning Model")

    left_column, right_column = st.columns(2, border=True)

    with left_column:
        st.badge("Bag Of Words (BOW) + Naive Bayes", icon=":material/looks_one:", color="blue")

        input_text = first_review.strip() if st.session_state.submit_review else second_review

        if input_text:
            cleaned_text = preprocess_text(input_text)
            bow = nb_vectorizer.transform([cleaned_text])
            pred = nb_model.predict(bow)[0]

            # Hitung confidence score
            confidence = None
            score_map = None

            if hasattr(nb_model, "predict_proba"):
                prob = nb_model.predict_proba(bow)[0]
                score_map = dict(zip(nb_model.classes_, prob))
                confidence = np.max(prob)  # Confidence = probabilitas tertinggi
            else:
                st.warning("Model tidak mendukung prediksi probabilitas.")

            # Tampilkan hasil prediksi dan confidence score
            if pred.upper() == "POSITIF":
                st.html(text_positive)
            elif pred.upper() == "NETRAL":
                st.html(text_netral)
            else:
                st.html(text_negative)

            st.markdown(f"**Confidence Score**: `{confidence:.4f}`" if confidence is not None else "Confidence tidak tersedia.")

            # Tampilkan visualisasi score per kelas jika tersedia
            if score_map:
                df_score = pd.DataFrame(list(score_map.items()), columns=["Sentimen", "Confidence"])
                st.bar_chart(df_score.set_index("Sentimen"))

            # Tampilkan highlight kata penting
            st.badge("Highlight Kata Kunci:", color="grey")
            feature_names = nb_vectorizer.get_feature_names_out()
            bow_vector = bow.toarray()[0]
            top_indices = np.argsort(bow_vector)[::-1][:5]
            top_words = [(feature_names[i], bow_vector[i]) for i in top_indices if bow_vector[i] > 0]

            if top_words:
                for word, weight in top_words:
                    st.markdown(f"`{word}` ({weight:.4f})")
            else:
                st.markdown("Tidak ada kata signifikan dari hasil BOW.")
        else:
            st.markdown("⚠️ Masukkan review terlebih dahulu.")


        


    with right_column:
        st.badge("TF-IDF + (SVM)", icon=":material/looks_two:", color="blue")

        input_text = first_review.strip() if st.session_state.submit_review else second_review

        if input_text:
            cleaned_text = preprocess_text(input_text)
            tfidf = svm_vectorizer.transform([cleaned_text])
            pred = svm_model.predict(tfidf)[0]

            if hasattr(svm_model, "decision_function"):
                scores = svm_model.decision_function(tfidf).flatten()
                confidence = np.max(scores)
            else:
                scores = None
                confidence = None

            if pred.upper() == "POSITIF":
                st.html(text_positive)
            elif pred.upper() == "NETRAL":
                st.html(text_netral)
            else:
                st.html(text_negative)
            
            st.markdown(f"**Confidence Score**: `{confidence:.4f}`" if confidence is not None else "Confidence tidak tersedia.")

            if scores is not None:
                st.bar_chart(dict(zip(svm_model.classes_, scores)))


            # Highlight Kata Kunci
            st.badge("Highlight Kata Kunci:", color="grey")
            feature_names = svm_vectorizer.get_feature_names_out()
            tfidf_vector = tfidf.toarray()[0]
            top_indices = np.argsort(tfidf_vector)[::-1][:5]
            top_words = [(feature_names[i], tfidf_vector[i]) for i in top_indices if tfidf_vector[i] > 0]

            for word, weight in top_words:
                st.markdown(f"`{word}` ({weight:.4f})")
        else:
            st.markdown("⚠️ Masukkan review terlebih dahulu.")

        


if selected == "Deep Learning Model":
    st.markdown("### Analisis Sentimen Menggunakan Deep Learning Model")

    left_column, right_column = st.columns(2, border=True)

    with left_column:
        st.badge("Word2Vec + Bi-LSTM", icon=":material/looks_one:", color="blue")

        input_text = first_review.strip() if st.session_state.submit_review else second_review

        if input_text:
            cleaned = preprocess_text(input_text)
            tokens = cleaned.split()
            doc_vector = document_vector(tokens, w2v_model)

            # Ubah bentuk agar sesuai input Bi-LSTM (1 sample, 1 timestep, n_features)
            doc_vector_reshaped = doc_vector.reshape((1, 1, -1))

            # Prediksi
            pred = bilstm_model.predict(doc_vector_reshaped, verbose=0)[0]

            # Decode hasil prediksi
            classes = bilstm_label_encoder.classes_
            predicted_index = np.argmax(pred)
            predicted_class = classes[predicted_index]
            confidence = pred[predicted_index] * 100 

            # Tampilkan hasil
            if predicted_class == "positif":
                st.html(text_positive)
            elif predicted_class == "netral":
                st.html(text_netral)
            else:
                st.html(text_negative)

            st.write(f"**Confidence Score:** `{confidence:.2f}%`")

            # Visualisasi probabilitas
            st.bar_chart(dict(zip(classes, pred)))

            # Highlight info
            st.badge("Highlight Kata Kunci:", color="grey")
            st.markdown("`highlight` tidak tersedia pada model deep learning")
        else:
            st.markdown("⚠️ Masukkan review terlebih dahulu.")


    with right_column:
        st.badge("FastText + Bi-GRU", icon=":material/looks_two:", color="blue")

        if input_text:
            cleaned_bigru = preprocess_text(input_text)
            tokens_bigru = cleaned_bigru.split()
            doc_vector_bigru = document_vector(tokens_bigru, bigru_fasttext)
            doc_vector_reshaped_bigru = doc_vector_bigru.reshape((1, 1, -1))  # (samples, timesteps, features)

            pred_bigru = bigru_model.predict(doc_vector_reshaped_bigru, verbose=0)[0]
            classes_bigru = bigru_label_encoder.classes_
            predicted_index_bigru = np.argmax(pred_bigru)
            predicted_class_bigru = classes_bigru[predicted_index_bigru]
            confidence_bigru = pred_bigru[predicted_index_bigru] * 100

            # Tampilkan visualisasi berdasarkan hasil
            if predicted_class_bigru == "positif":
                st.html(text_positive)
            elif predicted_class_bigru == "netral":
                st.html(text_netral)
            else:
                st.html(text_negative)
            
            st.markdown(f"**Confidence Score:**  `{confidence_bigru:.2f}%`")

            # Tampilkan bar chart
            st.bar_chart(dict(zip(classes_bigru, pred_bigru)))

            # Highlight info
            st.badge("Highlight Kata Kunci:", color="grey")
            st.markdown("`highlight` tidak tersedia pada model deep learning")
        else:
            st.markdown("⚠️ Masukkan review terlebih dahulu.")
