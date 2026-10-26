import os
import math
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ccxt  # Library untuk Binance/Bybit API

# ================= PENGATURAN HALAMAN =================
st.set_page_config(page_title="GPR Futures Scanner", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "300"))

# ================= DATA UNIVERSE (DAFTAR ASET) =================
# Fallback Daftar Saham BEI (IDX)
FALLBACK_IDX = """
BBCA BBRI BMRI TLKM ASII BBNI AMMN BYAN TPIA CPIN UNVR ICBP KLBF 
ADRO GGRM INTP PTBA MDKA PGAS BRPT BUMI ARTO MEDC TOWR AKRA EXCL 
ITMG INCO SMGR ANTM SIDO MYOR INDF UNTR MNCN SCM MAKA GOTO BREN
""".split()
FALLBACK_IDX = list(dict.fromkeys(FALLBACK_IDX))

# Fallback Daftar Saham US (Gotrade Ready / S&P 500 Top)
FALLBACK_US = """
AAPL MSFT NVDA AMZN GOOGL TSLA BRK-B LLY JPM V XOM UNH MA PG JNJ 
COST HD MRK ABBV CRM CVX AMD BAC NFLX KO PEP WMT TMO LIN MCD CSCO 
ACN ABT WFC DIS TXN ORCL INTU QCOM CAT AMGN DHR VZ IBM COP PM GE
""".split()

# Fallback Daftar Crypto Futures
CRYPTO_FUTURES_CODES = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "AVAX/USDT", "LINK/USDT", "DOT/USDT", "BCH/USDT", "LTC/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "INJ/USDT", "RENDER/USDT", "PEPE/USDT", "SUI/USDT", "TAO/USDT"
]

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_crypto_futures_universe():
    try:
        # Mengambil semua market USDT-M (Linear Futures) yang aktif di Binance
        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        markets = exchange.load_markets()
        symbols = [s for s, m in markets.items() if m.get('linear') and m.get('active') and m.get('quote') == 'USDT']
        if len(symbols) < 10:
            raise ValueError("Koin terlalu sedikit")
        return symbols, f"Binance USDT-M Futures Asli ({len(symbols)} pair); Real-time Data"
    except Exception as e:
        return CRYPTO_FUTURES_CODES.copy(), f"Fallback Kripto Teratas (Gagal load API: {str(e)})"

def get_universe(market: str):
    if market == "IDX":
        return FALLBACK_IDX.copy(), "Daftar Saham Bluechip BEI (Statis)"
    elif market == "US":
        return FALLBACK_US.copy(), "Daftar Saham Top US / S&P 500 (Statis)"
    else:
        return fetch_crypto_futures_universe()

# ================= FUNGSI PENGAMBILAN DATA (OHLCV) =================
def normalize_yf_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
    df = df.reset_index()
    aliases = {"Date": "time", "Datetime": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=aliases)
    if "close" not in df.columns: return pd.DataFrame()
    for col in ("open", "high", "low"):
        if col not in df.columns: df[col] = df["close"]
    if "volume" not in df.columns: df["volume"] = 0.0
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.sort_values("time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)

@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_history_cached(market: str, code: str, timeframe: str, cache_buster: int = 0) -> pd.DataFrame:
    # --- LOGIKA CRYPTO FUTURES (CCXT BINANCE) ---
    if market == "Crypto":
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True, 
                'options': {'defaultType': 'future'}
            })
            # Mapping timeframe untuk ccxt Binance
            tf_map = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
            ohlcv = exchange.fetch_ohlcv(code, tf_map[timeframe], limit=500)
            if not ohlcv: return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            return df
        except Exception as exc:
            raise RuntimeError(f"Binance API Error: {str(exc)[:130]}") from exc

    # --- LOGIKA SAHAM US & IDX (YAHOO FINANCE) ---
    else:
        interval_map = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
        period_map = {"5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}
        ticker = f"{code}.JK" if market == "IDX" else code
        try:
            raw = yf.Ticker(ticker).history(
                period=period_map.get(timeframe, "5y"),
                interval=interval_map.get(timeframe, "1d"),
                auto_adjust=False, actions=False, timeout=20,
            )
        except Exception as exc:
            raise RuntimeError(f"Yahoo Finance Error: {str(exc)[:130]}") from exc
        return normalize_yf_history(raw)

# ================= MATEMATIKA & INDIKATOR TEKNIKAL =================
def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def pine_rma(values, length):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if length <= 0 or len(values) < length: return out
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) < length: return out
    seed_end = int(valid_idx[:length][-1])
    out[seed_end] = float(np.mean(values[valid_idx[:length]]))
    alpha = 1.0 / length
    for i in range(seed_end + 1, len(values)):
        if np.isfinite(values[i]):
            prev = out[i - 1]
            if not np.isfinite(prev):
                prior = values[max(0, i-length+1):i+1]
                if np.isfinite(prior).sum() == length: out[i] = float(np.mean(prior))
            else:
                out[i] = alpha * values[i] + (1.0-alpha) * prev
    return out

def matern32_kernel(x, z, amplitude, scale):
    d = x - z
    distance2 = d[0]**2 + 0.8*d[1]**2 + 0.6*d[2]**2 + 0.8*d[3]**2 + 0.6*d[4]**2
    u = math.sqrt(3.0 * max(float(distance2), 0.0)) / scale
    return amplitude * (1.0 + u) * math.exp(-u)

# ================= ALGORITMA GAUSSIAN PROCESS REGRESSION (GPR) =================
def gpr_profile(timeframe):
    if timeframe in ("5m", "15m"):
        return dict(training=30, momentum_len=6, vol_len=24, ema_fast=8, ema_slow=26, kernel_scale=1.8, noise_ratio=0.85, volume_fade=0.35)
    if timeframe == "1h":
        return dict(training=35, momentum_len=5, vol_len=20, ema_fast=8, ema_slow=21, kernel_scale=1.6, noise_ratio=0.65, volume_fade=0.50)
    return dict(training=30, momentum_len=5, vol_len=14, ema_fast=8, ema_slow=21, kernel_scale=1.5, noise_ratio=0.30, volume_fade=0.65)

def gpr_project(df, horizon=10, training=30, momentum_len=5, vol_len=14, ema_fast=8, ema_slow=21, kernel_scale=1.5, noise_ratio=0.30, volume_fade=0.65, band_sigma=1.96, atr_width=0.70, validation_candles=3):
    min_needed = max(training + vol_len + ema_slow + 10, 60)
    idx = len(df) - 1 - validation_candles 
    
    if idx < min_needed:
        raise ValueError(f"Data candle kurang (Ada {len(df)}, butuh {min_needed + validation_candles})")
        
    close, volume = df["close"].astype(float).to_numpy(), df["volume"].fillna(0).astype(float).to_numpy()
    high, low = df["high"].astype(float).to_numpy(), df["low"].astype(float).to_numpy()

    # Ekstraksi Fitur (Features)
    logret = np.full(len(close), np.nan)
    logret[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))
    ret_series = pd.Series(logret)
    momentum = ret_series.ewm(span=momentum_len, adjust=False, min_periods=1).mean().to_numpy()
    volatility = ret_series.rolling(vol_len, min_periods=vol_len).std(ddof=0).to_numpy()
    vol_base = pd.Series(volatility).rolling(training, min_periods=training).mean().to_numpy()
    avg_volume = pd.Series(volume).rolling(vol_len, min_periods=vol_len).mean().to_numpy()
    
    rel_volume = np.zeros(len(close))
    valid = (volume > 0) & (avg_volume > 0) & np.isfinite(avg_volume)
    rel_volume[valid] = np.log(volume[valid] / avg_volume[valid])
    
    fast, slow = ema(pd.Series(close), ema_fast).to_numpy(), ema(pd.Series(close), ema_slow).to_numpy()
    direction = (fast - slow) / np.maximum(close, 1e-12)

    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    atr = pine_rma(tr, vol_len)

    vol_ref = max(float(vol_base[idx]) if np.isfinite(vol_base[idx]) else 0.0001, 0.0001)

    X, y = [], []
    for r in range(training):
        target_idx = idx - training + 1 + r
        feature_idx = target_idx - 1
        if feature_idx < 1 or not np.isfinite(logret[target_idx]): continue
        X.append([
            logret[feature_idx] / vol_ref, momentum[feature_idx] / vol_ref, rel_volume[feature_idx],
            volatility[feature_idx] / vol_ref - 1.0, direction[feature_idx] / vol_ref,
        ])
        y.append(logret[target_idx])

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    
    signal_var = max(float(np.mean(y*y)), 1e-8)
    noise_var = signal_var * noise_ratio + 1e-10
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            val = matern32_kernel(X[i], X[j], signal_var, kernel_scale)
            if i == j: val += noise_var
            K[i, j] = K[j, i] = val

    K += np.eye(n) * 1e-6
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        raise ValueError("Matriks GPR tidak stabil, lompati aset ini.")

    price_start = float(close[idx])
    price_prev, ret_prev, mom_prev = price_start, float(logret[idx]), float(momentum[idx])
    fast_prev, slow_prev = float(fast[idx]), float(slow[idx])
    volume_prev, vol_prev = float(rel_volume[idx]), float(volatility[idx])
    
    cumulative_return = cumulative_variance = 0.0
    predictions = []
    total_steps = horizon + validation_candles 

    for h in range(total_steps):
        fade = volume_fade ** h
        assumed_vol = vol_ref + (vol_prev - vol_ref) * fade
        x_star = np.array([
            ret_prev / vol_ref, mom_prev / vol_ref, volume_prev * fade,
            assumed_vol / vol_ref - 1.0, ((fast_prev - slow_prev) / max(price_prev, 1e-12)) / vol_ref,
        ])
        kstar = np.array([matern32_kernel(X[i], x_star, signal_var, kernel_scale) for i in range(n)])
        pred_ret = float(kstar @ alpha)
        try:
            solved = np.linalg.solve(L, kstar)
            predictive_variance = max(float(signal_var + noise_var - solved @ solved), 1e-12)
        except: predictive_variance = 1e-12
            
        cumulative_variance += predictive_variance
        cumulative_return += pred_ret
        price_pred = price_start * math.exp(cumulative_return)
        sigma_cum = math.sqrt(max(cumulative_variance, 0.0))
        
        # Batas Atas dan Bawah (Pita GPR / ATR)
        distance = atr_width * float(atr[idx]) * math.sqrt(h + 1.0)
        upper, lower = price_pred + distance, max(1e-12, price_pred - distance)
            
        real_idx = idx + h + 1
        actual_price = float(close[real_idx]) if real_idx < len(df) else None

        predictions.append({
            "step": h + 1, "is_future": actual_price is None,
            "price": round(price_pred, 6), "actual_price": round(actual_price, 6) if actual_price is not None else None,
            "upper": round(upper, 6), "lower": round(lower, 6)
        })
        
        # Update previous state for autoregression
        price_prev, ret_prev = price_pred, pred_ret
        a_m = 2.0 / (momentum_len + 1.0)
        mom_prev = a_m * pred_ret + (1.0-a_m) * mom_prev
        a_f, a_s = 2.0 / (ema_fast + 1.0), 2.0 / (ema_slow + 1.0)
        fast_prev = a_f * price_pred + (1.0-a_f) * fast_prev
        slow_prev = a_s * price_pred + (1.0-a_s) * slow_prev

    latest_actual_price = float(close[-1])
    trend = "BULLISH" if fast[-1] > slow[-1] else "BEARISH"
    mom_now = float(momentum[-1]) if np.isfinite(momentum[-1]) else 0.0
    
    sigma_cum_final = math.sqrt(max(cumulative_variance, 0.0))
    uncertainty = 100.0 * sigma_cum_final / (abs(cumulative_return) + sigma_cum_final + 1e-10)
    
    # Validation Logic (Menghitung Hit Rate & MAPE T-3)
    validation_errors = []
    hits = 0
    for p in predictions:
        if not p["is_future"] and p["actual_price"] is not None:
            err = abs(p["price"] - p["actual_price"]) / max(p["actual_price"], 1e-12)
            validation_errors.append(err)
            if p["lower"] <= p["actual_price"] <= p["upper"]: hits += 1
                
    val_count = len(validation_errors)
    validation_mape = (sum(validation_errors) / val_count) if val_count > 0 else 1.0
    validation_hit_rate = (hits / val_count) if val_count > 0 else 0.0

    return_from_now = (predictions[-1]["price"] / latest_actual_price - 1.0) * 100.0

    return {
        "price": round(latest_actual_price, 6), "projection_start_price": round(price_start, 6),
        "trend": trend, "momentum": "POSITIVE" if mom_now > 0 else "NEGATIVE",
        "gpr_return_pct": round(return_from_now, 3), "uncertainty_pct": round(uncertainty, 2),
        "validation_mape": validation_mape, "validation_hit_rate": validation_hit_rate,
        "predictions": predictions
    }

def analyze_one(market: str, code: str, timeframe: str, horizon: int) -> dict:
    hist = get_history_cached(market, code, timeframe, int(time.time() // CACHE_SECONDS))
    if hist.empty: raise ValueError("Data historis kosong/tidak tersedia.")
        
    params = gpr_profile(timeframe)
    result = gpr_project(hist, horizon=horizon, validation_candles=3, **params)
    result["code"], result["market"] = code, market
    
    # Scoring System
    hit_score = result["validation_hit_rate"] * 30
    mape_score = max(0, 20 - (result["validation_mape"] * 100 * 4))
    trend_score = 15 if result["trend"] == "BULLISH" else 0
    mom_score = 10 if result["momentum"] == "POSITIVE" else 0
    ret_score = 25 * max(0, min(abs(result["gpr_return_pct"]) / 5.0, 1.0)) # Absolute karena Shorting diperbolehkan
    
    score = max(0, min(100, hit_score + mape_score + trend_score + mom_score + ret_score))
    result["score"] = round(score, 1)
    
    # Sinyal Futures (LONG atau SHORT)
    if result["trend"] == "BULLISH" and result["gpr_return_pct"] > 0 and result["validation_hit_rate"] >= 0.66:
        result["signal"] = "🟢 KANDIDAT LONG"
    elif result["trend"] == "BEARISH" and result["gpr_return_pct"] < 0 and result["validation_hit_rate"] >= 0.66:
        result["signal"] = "🔴 KANDIDAT SHORT"
    else:
        result["signal"] = "⚪ PANTAU / HINDARI"
        
    return result

# ================= TAMPILAN ANTARMUKA (UI) STREAMLIT =================
st.markdown("""
<style>
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1400px;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;}
[data-testid="stMetric"] {background:#1E293B; border:1px solid #334155; padding:15px; border-radius:10px;}
div.stButton > button {border-radius:8px; min-height:45px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 GPR Futures & Market Scanner")
st.markdown("Cari peluang **LONG** dan **SHORT** menggunakan *Machine Learning Gaussian Process Regression (GPR)*.")

with st.sidebar:
    st.header("⚙️ Konfigurasi Scanner")
    market = st.selectbox("Pasar (Market)", ["Crypto", "US", "IDX"], format_func=lambda x: {"Crypto":"⚡ Crypto Futures (Binance)", "US":"🇺🇸 Saham Amerika", "IDX":"🇮🇩 Saham Indonesia (BEI)"}[x])
    tf_label = st.selectbox("Timeframe", ["15 menit", "5 menit", "1 jam", "Harian"], index=0)
    timeframe = {"Harian":"1d", "1 jam":"1h", "15 menit":"15m", "5 menit":"5m"}[tf_label]
    horizon = st.selectbox("Proyeksi Kedepan (Candle)", [3, 5, 10, 20], index=2)
    
    st.divider()
    batch_size = st.select_slider("Jumlah Scan per Batch", options=[20, 50, 100, 200], value=50)
    auto_scan = st.checkbox("Scan Otomatis sampai Selesai", value=False)
    min_score = st.slider("Filter Skor Minimal", 40, 90, 70, 5)
    
    st.divider()
    start_new = st.button("↻ Reset & Mulai Baru", use_container_width=True)
    scan_next = st.button("▶ Mulai / Lanjutkan Scan", type="primary", use_container_width=True)
    if st.button("🗑️ Hapus Cache (Refresh Data)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

codes, universe_source = get_universe(market)
target_count = len(codes)

state_key = f"{market}_{timeframe}_{horizon}"
if "scanner_states" not in st.session_state:
    st.session_state.scanner_states = {}
if state_key not in st.session_state.scanner_states or start_new:
    st.session_state.scanner_states[state_key] = {"cursor":0, "results":[], "errors":[]}
    if start_new: st.rerun()

state = st.session_state.scanner_states[state_key]

# HEADER METRICS
col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Total Aset di Market", target_count)
col2.metric("Telah di Scan", f"{state['cursor']} / {target_count}")
col3.markdown(f"**Sumber Data:** {'Binance API (Real-time)' if market == 'Crypto' else 'Yahoo Finance (Delay)'} <br> *{universe_source}*", unsafe_allow_html=True)

# THREAD WORKER
def _worker_task(code):
    time.sleep(0.1 if market == "Crypto" else 0.05) # Rate limit protection
    try:
        row = analyze_one(market, code, timeframe, int(horizon))
        return "SUCCESS", code, row
    except Exception as exc:
        return "ERROR", code, str(exc)[:150]

# EKSEKUSI SCAN
if scan_next:
    if state["cursor"] >= target_count:
        st.info("Seluruh aset di market ini sudah selesai di-scan. Silakan Reset untuk mengulang.")
    else:
        step_size = target_count if auto_scan else int(batch_size)
        progress = st.progress(0.0, text="Mempersiapkan Scanner...")
        status = st.empty()
        
        # Crypto batasi 4 thread agar tidak kena IP Ban (HTTP 429) dari Binance
        max_threads = 4 if market == "Crypto" else 8

        while state["cursor"] < target_count:
            batch_codes = codes[state["cursor"] : state["cursor"] + step_size]
            known = {x["code"] for x in state["results"]} | {x["code"] for x in state["errors"]}
            to_process = [c for c in batch_codes if c not in known]

            if to_process:
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = {executor.submit(_worker_task, code): code for code in to_process}
                    for future in as_completed(futures):
                        status_type, code, payload = future.result()
                        if status_type == "SUCCESS":
                            state["results"].append(payload)
                        else:
                            state["errors"].append({"code": code, "error": payload})
                        
                        state["cursor"] += 1
                        pct = min(state["cursor"] / target_count, 1.0)
                        progress.progress(pct, text=f"Sedang menganalisis {code} ({state['cursor']}/{target_count})")
            else:
                state["cursor"] = min(state["cursor"] + step_size, target_count)

            # Sortir by Score lalu Absolute Return (Baik Long maupun Short)
            state["results"].sort(key=lambda x: (x["score"], abs(x["gpr_return_pct"])), reverse=True)
            if not auto_scan: break

        progress.empty()
        status.success(f"✅ Scan selesai! Diproses: {state['cursor']} koin.")

# DATA PREPARATION
results = state["results"]
errors = state["errors"]
qualified = [x for x in results if float(x.get("score", 0)) >= min_score]
candidates = [x for x in qualified if "KANDIDAT" in x.get("signal", "")]

st.divider()

# TABS TAMPILAN
tab_ranking, tab_detail, tab_status = st.tabs(["🏆 Hasil Ranking Futures", "📈 Detail Chart GPR", "🧾 Status / Error Log"])

with tab_ranking:
    st.subheader(f"Daftar Kandidat Trading ({len(candidates)} Ditemukan)")
    if qualified:
        table = []
        for rank, item in enumerate(qualified, 1):
            table.append({
                "Rank": rank, 
                "Aset": item["code"],
                "Sinyal": item["signal"], 
                "Skor Validasi": f"{item['score']} / 100", 
                "Akurasi T-3": f"{int(item['validation_hit_rate']*100)}%",
                "Proyeksi GPR": f"{item['gpr_return_pct']:+.2f}%",
                "Harga Saat Ini": item["price"]
            })
        df_table = pd.DataFrame(table)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada aset yang lolos kriteria. Silakan tekan tombol Mulai Scan.")

with tab_detail:
    if results:
        selected_asset = st.selectbox("Pilih aset untuk dianalisis lebih dalam:", [x["code"] for x in results])
        item = next(x for x in results if x["code"] == selected_asset)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Harga Terkini", f"{item['price']:,.4f}")
        m2.metric("Sinyal Sistem", item["signal"])
        m3.metric("Estimasi P/L (Profit)", f"{item['gpr_return_pct']:+.2f}%")
        m4.metric("Risk Uncertainty", f"{item.get('uncertainty_pct', float('nan')):.2f}%")
        
        preds = item.get("predictions", [])
        if preds:
            st.markdown(f"#### Proyeksi Pergerakan Harga ({selected_asset})")
            chart_data = [{"Candle": "Start (T-3)", "GPR": item["projection_start_price"], "Harga Asli": item["projection_start_price"], "Batas Atas": item["projection_start_price"], "Batas Bawah": item["projection_start_price"]}]
            valid_count = sum(1 for p in preds if not p["is_future"])
            
            for p in preds:
                lbl = f"Future +{p['step'] - valid_count}" if p["is_future"] else (f"Valid T-{valid_count - p['step']}" if valid_count - p['step'] > 0 else "Hari Ini (T0)")
                chart_data.append({
                    "Candle": lbl, 
                    "GPR": p["price"], 
                    "Harga Asli": p["actual_price"], 
                    "Batas Atas": p["upper"], 
                    "Batas Bawah": p["lower"]
                })
            
            st.line_chart(pd.DataFrame(chart_data).set_index("Candle"))
            
            # Tips Penempatan Posisi
            is_long = "LONG" in item["signal"]
            is_short = "SHORT" in item["signal"]
            if is_long or is_short:
                stop_loss = preds[-1]['lower'] if is_long else preds[-1]['upper']
                target = preds[-1]['price']
                st.success(f"💡 **Trading Plan (Referensi):** Jika Anda mengambil posisi **{item['signal'].replace('KANDIDAT ', '')}**, Anda bisa mempertimbangkan **Take Profit (TP)** di area `{target:,.4f}` dan **Stop Loss (SL)** di sekitar area `{stop_loss:,.4f}` (berdasarkan deviasi GPR).")
            else:
                st.warning("⚠️ Algoritma tidak menyarankan entri (Market sedang bingung/Choppy). Lebih baik cari koin lain.")
    else:
        st.write("Silakan scan data terlebih dahulu.")

with tab_status:
    st.write(f"Total Berhasil: **{len(results)}** | Gagal (Error): **{len(errors)}**")
    if errors:
        st.dataframe(pd.DataFrame(errors), use_container_width=True)
    else:
        st.success("Semua aset berhasil ditarik tanpa error.")
