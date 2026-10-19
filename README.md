# DataSphere — Streamlit + MongoDB

## Menjalankan di VS Code (Windows)
1. Ekstrak folder dan buka di VS Code.
2. Buka Terminal → New Terminal.
3. Jalankan:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang ditampilkan Streamlit, biasanya http://localhost:8501.

## MongoDB Atlas (opsional)
Buat cluster, database user, dan izinkan IP komputer pada Network Access. Salin connection string dari Connect → Drivers. Lalu di PowerShell:

```powershell
$env:MONGODB_URI="mongodb+srv://USERNAME:PASSWORD@CLUSTER_URL/?retryWrites=true&w=majority"
$env:MONGODB_DB="datasphere"
streamlit run app.py
```

Ganti URI dengan milik sendiri. Jangan unggah password atau file secrets ke GitHub. Jika URI tidak diatur, aplikasi berjalan dengan penyimpanan sementara di session lokal.

## Modul
- Home
- Dataset Explorer (Diabetes, Iris, Wine, Breast Cancer)
- Machine Learning Lab (regression, classification, K-Means clustering)
- MongoDB & Retraining
- Community
- About & Methodology

Catatan: retraining saat ini adalah demo baseline pada dataset Diabetes bawaan; record formulir belum otomatis menjadi data latih karena skema fitur dan target berbeda. Sentiment analysis dan like/dislike persisten belum diimplementasikan.
