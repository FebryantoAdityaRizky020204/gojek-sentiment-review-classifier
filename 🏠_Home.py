import streamlit as st
import pandas as pd
import base64
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="HOME - Gojek Statement Review Classifier",
    page_icon="🏠",
    layout="wide"
)


# ==================================================================================
SVM_MODEL_PATH = ".\models\PREPROCESS\svm_tfidf_model_20250605_042710.pkl"
VECTORIZER_PATH = ".\models\PREPROCESS\svm_tfidf_vectorizer_20250605_042710.pkl"

with open(SVM_MODEL_PATH, 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vec_file:
    svm_vectorizer = pickle.load(vec_file)

# === Preprocessing Function ===
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize

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

text_negative = "<p style='font-weight:bold;font-size:30px'>&#9593; HASIL: <span style='color:red;'>NEGATIVE</span> &#9595;</p>"
text_netral = "<p style='font-weight:bold;font-size:30px'>&#9599; HASIL: <span style='color:grey;'>NETRAL</span> &#9599;</p>"
text_positive = "<p style='font-weight:bold;font-size:30px'>&#9595; HASIL: <span style='color:green;'>POSITIVE</span> &#9593;</p>"

data_dummy_negative = {"positive": 30,"neutral": 20,"negative": 50}
data_dummy_netral = {"positive": 30,"neutral": 50,"negative": 20}
data_dummy_positive = {"positive": 70,"neutral": 20,"negative": 10}


if selected == "Traditional Machine Learning Model":
    st.markdown("### Analisis Sentimen Menggunakan Traditional Machine Learning Model")

    left_column, right_column = st.columns(2, border=True)

    with left_column:
        st.badge("Bag Of Words (BOW) + Naive Bayes", icon=":material/looks_one:", color="blue")
        # st.markdown("#### Bag Of Words (BOW) + Naive Bayes")
        st.html(text_negative)
        st.bar_chart(data_dummy_negative, stack=False)
        # st.markdown("---")
        st.badge("Highlight Kata Kunci:", color="grey")
        st.markdown("`adadeh` mau tau aja `kamu`")

    with right_column:
        st.badge("TF-IDF + (SVM)", icon=":material/looks_two:", color="blue")
        # st.markdown("#### TF-IDF + (SVM)")
        st.html(text_positive)
        st.bar_chart(data_dummy_positive, stack=False)

        st.badge("Highlight Kata Kunci:", color="grey")
        st.markdown("kalian `sabar` dulu ya")
        


if selected == "Deep Learning Model":
    st.markdown("### Analisis Sentimen Menggunakan Deep Learning Model")

    left_column, right_column = st.columns(2, border=True)

    with left_column:
        st.badge("Word2Vec + Bi-LSTM", icon=":material/looks_one:", color="blue")
        # st.markdown("#### Word2Vec + Bi-LSTM")
        st.html(text_positive)
        st.bar_chart(data_dummy_positive, stack=False)

        st.badge("Highlight Kata Kunci:", color="grey")
        st.markdown("nanti kalau `sudah` kubilang di `grup`")

    with right_column:
        st.badge("FastText + Bi-GRU", icon=":material/looks_two:", color="blue")
        # st.markdown("#### FastText + Bi-GRU")
        st.html(text_netral)
        st.bar_chart(data_dummy_netral, stack=False)

        st.badge("Highlight Kata Kunci:", color="grey")
        st.markdown("lagi `males` malam `nanti` lagi")