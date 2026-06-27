import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import confusion_matrix

# ======================
# LOAD MODEL
# ======================
try:
    rf = joblib.load("model_rf.pkl")
    logreg = joblib.load('model_logreg.pkl') 
    ann = joblib.load('model_ann.pkl') 
    scaler = joblib.load("scaler.pkl")
    fitur = joblib.load("fitur.pkl")
    # TAMBAHAN
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
except:
    X_test, y_test = None, None
    st.error("Model atau file tidak ditemukan!")
    st.stop()

# ======================
# CONFIG UI
# ======================
st.set_page_config(page_title="Prediksi Kredit", layout="wide")
st.markdown("""
<style>

/* =========================
   BACKGROUND UTAMA
========================= */
.stApp {
    background-color: #f8fafc;
    color: #111;
}

/* =========================
   SIDEBAR (KUNING ORANGE)
========================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffb347, #ffcc33);
    color: #111;
}

/* Text sidebar */
section[data-testid="stSidebar"] * {
    color: #111 !important;
    font-weight: 500;
}

/* =========================
   BUTTON
========================= */
.stButton>button {
    background: linear-gradient(90deg, #f7971e, #ffd200);
    color: #111;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}
div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #f7971e, #ffd200) !important;
}

/* Hover button */
.stButton>button:hover {
    background: linear-gradient(90deg, #e68900, #ffcc00);
    color: black;
}

/* =========================
   INPUT BOX (KUNING SOFT)
========================= */
input, select {
    background-color: #fff8cc !important;
    color: #111 !important;
    border-radius: 8px !important;
    border: 1px solid #ffd200 !important;
}

/* =========================
   METRIC BOX
========================= */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #eee;
}

/* =========================
   PROGRESS BAR
========================= */
div[data-testid="stProgressBar"] > div {
    background-color: #f7971e !important;
}

/* =========================
   HEADER TEXT
========================= */
h1, h2, h3 {
    color: #111 !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #f7971e, #ffd200);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    ">
        <h1 style="color:white; text-align:center;">
            💳 Dashboard Prediksi Risiko Kredit
        </h1>
        <p style="color:white; text-align:center; font-size:18px;">
            Aplikasi untuk memprediksi kemungkinan gagal bayar
        </p>
    </div>
""", unsafe_allow_html=True)
# ====================== 
# SIDEBAR 
# ====================== 
menu = st.sidebar.radio( "Menu", ["Prediction", "Model Comparison", "Feature Importance", "Confusion Matrix"] )

# ======================
# INPUT USER
# ======================
if menu == "Prediction":

    col1, col2 = st.columns([1, 1])  # kiri : kanan

    # ======================
    # KOLOM KIRI (INPUT)
    # ======================
    with col1:
        st.subheader("📥 Input Data")

        usia = st.number_input("Usia", 18, 100, 30)
        lama_kerja = st.number_input("Lama Bekerja (tahun)", 0, 50, 5)
        pendapatan = st.number_input("Pendapatan Tahunan", 0.0, 50000000.0)
        skor = st.number_input("Skor Kredit", 300, 850, 650)

        pekerjaan = st.selectbox("Status Pekerjaan", ["Bekerja", "Tidak"])
        tujuan = st.selectbox("Tujuan Pinjaman", ["Konsumtif", "Bisnis"])

        jumlah = st.number_input("Jumlah Pinjaman", 0.0, 10000000.0)
        hutang = st.number_input("Hutang Saat Ini", 0.0, 2000000.0)
        bunga = st.number_input("Suku Bunga (%)", 0.0, 10.0)
        dti = st.number_input("Rasio Hutang terhadap Pendapatan", 0.0, 0.3)

        prediksi_btn = st.button("🔍 Prediksi")

    # ======================
    # KOLOM KANAN (HASIL)
    # ======================
    with col2:
        st.subheader("📊 Hasil Prediksi")

        if prediksi_btn:
            try:
                pekerjaan_val = 1 if pekerjaan == "Bekerja" else 0
                tujuan_val = 1 if tujuan == "Bisnis" else 0

                input_dict = {
                    'usia': usia,
                    'status_pekerjaan': pekerjaan_val,
                    'lama_bekerja_tahun': lama_kerja,
                    'pendapatan_tahunan': pendapatan,
                    'skor_kredit': skor,
                    'hutang_saat_ini': hutang,
                    'tujuan_pinjaman': tujuan_val,
                    'jumlah_pinjaman': jumlah,
                    'suku_bunga': bunga,
                    'rasio_hutang_terhadap_pendapatan': dti,
                }

                input_dict.update({
                    'lama_riwayat_kredit_tahun': 5,
                    'aset_tabungan': pendapatan * 0.2,
                    'gagal_bayar_tercatat': 0,
                    'tunggakan_2thn_terakhir': 0,
                    'catatan_negatif': 0,
                    'tipe_produk': 1,
                    'rasio_pinjaman_terhadap_pendapatan': jumlah / pendapatan if pendapatan else 0,
                    'rasio_pembayaran_terhadap_pendapatan': (jumlah * bunga / 100) / pendapatan if pendapatan else 0,
                })

                data = pd.DataFrame(
                    [[input_dict.get(f, 0) for f in fitur]],
                    columns=fitur
                )

                data_scaled = scaler.transform(data)

                prob = rf.predict_proba(data_scaled)[0]
                lancar, gagal = prob
                pred = rf.predict(data_scaled)[0]

                if pred == 1:
                    st.error("❌ Berisiko Gagal Bayar")
                else:
                    st.success("✅ Kredit Lancar")

                st.write(f"Probabilitas Lancar: **{lancar:.2%}**")
                st.write(f"Probabilitas Gagal: **{gagal:.2%}**")
                # ======================
                # VISUALISASI RISIKO
                # ======================

                st.subheader("📊 Risk Score")

                # Progress bar (utama)
                st.write("Gagal Bayar")
                st.progress(float(gagal))

                # Angka risk score
                st.metric("Risk Score", f"{gagal:.2f}")

                # Bar chart perbandingan
                risk_df = pd.DataFrame({
                    "Kategori": ["Lancar", "Gagal"],
                    "Probabilitas": [lancar, gagal]
                })

                fig, ax = plt.subplots()

                ax.bar(
                    risk_df["Kategori"],
                    risk_df["Probabilitas"]
                )

                # Label persentase
                for i, v in enumerate(risk_df["Probabilitas"]):
                    ax.text(i, v + 0.01, f"{v:.2%}", ha='center')

                ax.set_ylabel("Probabilitas")
                ax.set_title("Perbandingan Risiko")

                st.pyplot(fig)

                # ======================
                # LEVEL RISIKO
                # ======================

                if gagal > 0.7:
                    st.error("🔥 Risiko Tinggi")
                elif gagal > 0.4:
                    st.warning("⚠️ Risiko Sedang")
                else:
                    st.success("🟢 Risiko Rendah")

            except Exception as e:
                st.error(f"Error: {e}")
# ======================
# 2. MODEL COMPARISON
# ======================
elif menu == "Model Comparison":

    st.subheader("Perbandingan Model")

    if X_test is not None:
        acc_rf = rf.score(X_test, y_test)
        acc_log = logreg.score(X_test, y_test)
        acc_ann = ann.score(X_test, y_test)

        models = ["Random Forest", "LogReg", "ANN"]
        accuracy = [acc_rf, acc_log, acc_ann]

        fig, ax = plt.subplots()
        bars = ax.bar(models, accuracy)

        # Tambahin label
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha='center'
            )

        ax.set_title("Akurasi Model")
        ax.set_ylabel("Accuracy")

        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha='center')

        st.pyplot(fig)
    else:
        st.warning("Data test tidak tersedia")

# ======================
# 3. FEATURE IMPORTANCE
# ======================
elif menu == "Feature Importance":

    st.subheader("Feature Importance")

    importances = rf.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots()

    ax.barh(
        range(len(idx)),
        importances[idx]
    )

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([fitur[i] for i in idx])
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ======================
# 4. CONFUSION MATRIX
# ======================
elif menu == "Confusion Matrix":

    st.subheader("Confusion Matrix")

    if X_test is not None:
        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        cax = ax.matshow(cm)

        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)
    else:
        st.warning("Data test tidak tersedia")

