# 🚀 GPR Futures & Market Scanner

Aplikasi berbasis web (Streamlit) untuk memindai peluang *trading* di pasar **Crypto Futures (Binance USDT-M)**, **Saham US**, dan **Saham BEI (IDX)** menggunakan model **Gaussian Process Regression (GPR)**.

## ✨ Fitur Utama
- **Real-Time Data:** Integrasi langsung dengan `ccxt` untuk data Binance Futures tanpa delay.
- **Machine Learning GPR:** Proyeksi harga masa depan menggunakan kernel kustom *Matern 3/2*.
- **T-3 Validation:** Backtest otomatis 3 *candle* terakhir untuk mengecek hit-rate & akurasi model sebelum memberikan sinyal.
- **Sinyal Long/Short:** Penentuan otomatis kandidat **LONG** atau **SHORT** lengkap dengan estimasi SL & TP berbasis *Average True Range* (ATR).

## 🛠️ Cara Menjalankan di Komputer Lokal

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/USERNAME_ANDA/NAMA_REPO_ANDA.git](https://github.com/USERNAME_ANDA/NAMA_REPO_ANDA.git)
   cd NAMA_REPO_ANDA
