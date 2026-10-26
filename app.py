import streamlit as st
import ccxt

st.set_page_config(
    page_title="Crypto Futures Scanner",
    page_icon="🚀"
)

st.title("🚀 Crypto Futures Scanner")

st.success("Streamlit berhasil!")

st.write("CCXT version:", ccxt.__version__)

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    }
})

markets = exchange.load_markets()

st.success(f"Binance berhasil terhubung: {len(markets)} market")