import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load semua model & tools
vectorizer = joblib.load("vectorizer_tfidf_DPS.joblib")
model_sentimen = joblib.load("model_logistic_regression_DPS.joblib")
label_encoder = joblib.load("label_encoder_DPS.joblib")
emosi_lexicon = joblib.load("emosi_lexicon.joblib")

# Fungsi deteksi emosi
def deteksi_emosi(teks):
    for emosi, kata_kunci in emosi_lexicon.items():
        for kata in kata_kunci:
            if kata in teks:
                return emosi
    return "netral"

# Fungsi prediksi
def predict_sentiment(text):
    text_clean = text.lower().strip()
    text_vector = vectorizer.transform([text_clean])
    pred_sentimen_enc = model_sentimen.predict(text_vector)[0]
    pred_sentimen = label_encoder.inverse_transform([pred_sentimen_enc])[0]
    pred_emosi = deteksi_emosi(text_clean)
    return pred_sentimen, pred_emosi


# Layout Streamlit
st.set_page_config(page_title="Analisis Sentimen DPS", page_icon="🏛️", layout="wide")

with st.sidebar:
    menu = option_menu(
        menu_title=None,
        options=["🏠 Home", "🎯 Halaman Hasil Model", "📈 Halaman Sentimen"],
        icons=[],
        default_index=0,
        styles={
            "container": {
                "background-color": "transparent",
                "padding": "0px",
            },
            "icon": {"display": "none"}, 
            "nav-link": {
                "font-size": "16px",
                "color": "#ccc",
                "text-align": "left",
                "margin": "6px 0",
                "border-radius": "8px",
            },
            "nav-link:hover": {"background-color": "#333342"},
            "nav-link-selected": {
                "background-color": "#44445a",
                "font-weight": "bold",
                "color": "white",
            },
        },
    )

# 1. Home  
if menu == "🏠 Home":
    st.title("🏛️ Sistem Analisis Sentimen Pengguna Aplikasi DPS - Denpasar Prama Sewaka")
    st.markdown(
        """
        Author: Putu Mahdalika
        <style>
        .block-container {
            max-width: 950px;
            padding-top: 6rem;
            padding-bottom: 6rem;
            margin: auto;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Deskripsi aplikasi
    st.markdown(
        """
        Selamat Datang di Sistem Analisis Sentimen Ulasan Pengguna Aplikasi DPS. Sistem ini dibuat menggunakan bahasa pemrograman Python 🐍 dan framework GUI Streamlit. Data yang digunakan saya ambil dari Google Play mulai dari bulan Maret 2016 sampai dengan Juni 2025.
        """
        )

    # Menu yang tersedia
    st.subheader("Menu yang tersedia")

    st.markdown(
        """
        - 🎯**Halaman Hasil Model** adalah menu yang menampilkan hasil evaluasi model Logistic Regression berupa Accuracy, Precision, Recall, F1-Score.
        - 📈 **Halaman Sentimen** adalah menu kedua untuk melihat distribusi sentimen dan emosi, serta input analisis teks atau file CSV.  
        """
    )

    # Tentang aplikasi
    st.subheader("Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini dirancang untuk memprediksi sentimen dari ulasan yang diinputkan pengguna menggunakan beberapa model yang telah dilatih menggunakan dataset yang ada. Pengguna dapat memasukkan teks komentar, dan sistem akan memproses teks tersebut untuk menentukan sentimennya berdasarkan model tersebut.

        Cara Kerja Sistem:

        1. **Input Komentar**: Pengguna memasukkan teks komentar ke dalam sistem.
        2. **Prediksi Sentimen**: Teks atau file CSV yang sudah diproses kemudian dianalisis menggunakan model Logistic Regression yang sudah dilatih sebelumnya pada dataset untuk memprediksi sentimen.
        3. **Output Sentimen**: Hasil prediksi dari model (positif, negatif) dan emosi ditampilkan kepada pengguna.
        """
    )

# 2. Hasil Model
elif menu == "🎯 Halaman Hasil Model":
    st.title("🎯 Hasil Validasi Model")
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 950px;
            padding-top: 6rem;
            padding-bottom: 6rem;
            margin: auto;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Halaman ini berisikan tampilan hasil validasi dan evaluasi dari model yang telah dibuat dan yang akan digunakan pada halaman sentimen. Parameter validasi untuk model Logistic Regression ini berupa **Accuracy, Precision, Recall, dan F1-Score**.
        """
    )

    # Tombol untuk load hasil
    if st.button("📊 Lihat Hasil"):
        try:
            # Load hasil evaluasi model dari joblib
            metrics = joblib.load("hasil_evaluasi_logistic_regression.joblib")
            st.success("Berhasil memuat hasil evaluasi model!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
            col4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
            
            st.subheader("Confusion Matrix")

            # Load model & encoder
            model = joblib.load("model_logistic_regression_DPS.joblib")
            le = joblib.load("label_encoder_DPS.joblib")

            # Load vectorizer & data test
            tfidf = joblib.load("vectorizer_tfidf_DPS.joblib")
            df_test = pd.read_csv("data_test_DPS.csv")

            X_test_text = df_test['stopword_str']
            y_test = df_test['sentiment']
            X_test = tfidf.transform(X_test_text)

            # Prediksi
            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, cmap="Blues", ax=ax
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memuat hasil evaluasi: {e}")

    
# 3. Halaman Sentimen
elif menu == "📈 Halaman Sentimen":
    st.title("📈 Analisis Sentimen & Emosi")
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 950px;
            padding-top: 6rem;
            padding-bottom: 6rem;
            margin: auto;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Komentar", "File CSV"])

    # Input teks
    with tab1:
        input_text = st.text_area("Masukkan komentar anda:", height=100)
        if st.button("Prediksi Komentar"):
            if input_text.strip():
                sentimen, emosi = predict_sentiment(input_text)
                st.success(f"✅ Sentimen: {sentimen.lower()}")
                st.info(f"😀 Emosi: {emosi}")
            else:
                st.warning("Masukkan teks terlebih dahulu!")

    # Input CSV
    with tab2:
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if "text" in df.columns:
                teks_kolom = "text"
            elif "komentar" in df.columns:
                teks_kolom = "komentar"
            else:
                st.error("CSV harus punya kolom **text** atau **komentar**.")
                teks_kolom = None

            if teks_kolom:
                hasil = []
                for teks in df[teks_kolom].astype(str):
                    sentimen, emosi = predict_sentiment(teks)
                    hasil.append([teks, sentimen, emosi])

                df_hasil = pd.DataFrame(hasil, columns=["Teks", "Sentimen", "Emosi"])
                st.dataframe(df_hasil)

                # Grafik distribusi
                st.write("### 📊 Distribusi Sentimen")
                st.bar_chart(df_hasil["Sentimen"].value_counts())

                # Download
                csv = df_hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil Analisis",
                    data=csv,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv"
                )
