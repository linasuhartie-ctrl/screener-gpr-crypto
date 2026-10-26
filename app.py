import streamlit as st
import ccxt

st.set_page_config(
    page_title="Crypto Futures Scanner",
    page_icon="🚀"
)

st.title("🚀 Crypto Futures Scanner")

st.success("Aplikasi berhasil berjalan!")

st.write("CCXT version:", ccxt.__version__)