"""
GPR FUTURES INTELLIGENCE STRICT - Bybit USDT Perpetual Screener (Streamlit)
Jalankan lokal : streamlit run app.py
Engine         : gpr_bybit_screener.py (harus satu folder dengan file ini)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import gpr_bybit_screener as g

st.set_page_config(page_title="GPR Strict Screener | Bybit Perp", page_icon="📡", layout="wide")

TF_LABELS = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240"}
ACTIVE = ["INIT SL", "BE LOCKED", "LVL1 LOCKED"]
WIB = "Asia/Jakarta"
LOCKED_COLS = [("signal", "-"), ("state", "-"), ("age", np.nan), ("entry", np.nan), ("sl", np.nan),
               ("tp1", np.nan), ("tp2", np.nan), ("tp3", np.nan)]


# ----------------------------------------------------------------------------
# SCAN
# ----------------------------------------------------------------------------
def run_scan(tf, cfg, top, min_turn_m, symbols, workers, base):
    g.BASE = base
    session = requests.Session()
    session.headers["User-Agent"] = "gpr-screener/1.0"
    uni = g.universe(session, min_turn_m * 1e6, top, symbols or None)
    if uni.empty:
        raise RuntimeError("Universe kosong. Cek filter turnover / nama simbol.")
    results, fails = [], 0
    bar = st.progress(0.0, text=f"Scan {len(uni)} simbol...")
    min_bars = max(cfg.ema_slow + 50, 260)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(g.scan_symbol, session, r, tf, cfg, 1000, min_bars) for r in uni.to_dict("records")]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    results.append(r)
                else:
                    fails += 1
            except Exception:                                   # noqa: BLE001
                fails += 1
            bar.progress(k / len(futs), text=f"Scan {k}/{len(futs)}")
    bar.empty()
    return dict(raw=results, fails=fails, tf=tf, base=base, cfg=cfg,
                time=pd.Timestamp.now(tz=WIB).strftime("%d %b %Y %H:%M:%S WIB"))


def prepare(raw, max_age, choch_age, choch_dir):
    df = pd.DataFrame(raw)
    df["best"] = df[["l_score", "s_score"]].max(axis=1)
    df["bias"] = np.where(df["l_score"] >= df["s_score"], "LONG", "SHORT")
    fresh = df["signal"].ne("-") & (df["age"] <= max_age)
    active = df["state"].isin(ACTIVE)
    stale = df["signal"].ne("-") & ~fresh & ~active
    for col, empty in LOCKED_COLS:                              # sinyal lama & sudah closed disembunyikan
        df.loc[stale, col] = empty
    ch_any = df["choch"].ne("-") & (df["choch_age"] <= choch_age)
    for col, empty in [("choch", "-"), ("choch_age", np.nan), ("choch_lvl", np.nan)]:
        df.loc[~ch_any, col] = empty
    ch_sel = ch_any.copy()
    if choch_dir != "Semua":
        ch_sel &= df["choch"].eq(choch_dir.upper())
    df["is_sig"] = fresh | active
    df["is_fresh"] = fresh
    df["is_choch"] = ch_sel
    return df


# ----------------------------------------------------------------------------
# TAMPILAN TABEL
# ----------------------------------------------------------------------------
def color_dir(v):
    if isinstance(v, str):
        if v.startswith(("LONG", "BULL")):
            return "color:#16a34a;font-weight:600"
        if v.startswith(("SHORT", "BEAR")):
            return "color:#dc2626;font-weight:600"
    return ""


def show_table(view, color_cols):
    sty = view.style
    painter = getattr(sty, "map", None) or sty.applymap
    sty = painter(color_dir, subset=[c for c in color_cols if c in view.columns])
    st.dataframe(sty, width="stretch", hide_index=True)


def age_txt(a):
    return "" if pd.isna(a) else f"-{int(a)}b"


def base_cols(d):
    return {
        "Symbol": d["symbol"],
        "Price": d["price"].map(g.fmt_price),
        "24h%": d["chg24"].map("{:+.1f}".format),
        "Vol$M": d["turn"].map("{:.0f}".format),
    }


def view_signals(d):
    v = base_cols(d)
    v.update({
        "Signal": d["signal"] + " " + d["age"].map(age_txt),
        "State": d["state"],
        "Entry": d["entry"].map(g.fmt_price), "SL": d["sl"].map(g.fmt_price),
        "TP1": d["tp1"].map(g.fmt_price), "TP2": d["tp2"].map(g.fmt_price), "TP3": d["tp3"].map(g.fmt_price),
        "L/S": d["l_score"].map("{:.0f}".format) + "/" + d["s_score"].map("{:.0f}".format),
        "Trend": d["trend"], "MTF": d["mtf"], "GPR": d["gpr"],
        "GPR%": d["gpr_fc"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}"),
        "Pump": d["pump"], "Fund%": d["fund"].map("{:+.3f}".format),
    })
    return pd.DataFrame(v)


def view_choch(d):
    v = base_cols(d)
    v.update({
        "CHoCH": d["choch"] + " " + d["choch_age"].map(age_txt),
        "Level": d["choch_lvl"].map(g.fmt_price), "Liq": d["liq"],
        "Trend": d["trend"], "Str": d["struct"], "MTF": d["mtf"],
        "RSI": d["rsi"].map("{:.0f}".format), "GPR": d["gpr"],
        "L/S": d["l_score"].map("{:.0f}".format) + "/" + d["s_score"].map("{:.0f}".format),
        "Seq": d["seq"], "Pump": d["pump"],
    })
    return pd.DataFrame(v)


def view_scores(d):
    v = base_cols(d)
    v.update({
        "Bias": d["bias"], "Best": d["best"].map("{:.0f}".format),
        "L/S": d["l_score"].map("{:.0f}".format) + "/" + d["s_score"].map("{:.0f}".format),
        "Trend": d["trend"], "Str": d["struct"], "MTF": d["mtf"], "Seq": d["seq"], "Liq": d["liq"],
        "RSI": d["rsi"].map("{:.0f}".format), "Tok": d["token"], "GPR": d["gpr"], "Pump": d["pump"],
    })
    return pd.DataFrame(v)


# ----------------------------------------------------------------------------
# CHART
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_klines(symbol, tf, base):
    g.BASE = base
    return g.klines(requests.Session(), symbol, tf, 1000)


def make_chart(row, tf, base):
    df = load_klines(row["symbol"], tf, base)
    if df is None or len(df) < 30:
        return None
    c = df["close"].to_numpy()
    t = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(WIB)
    n = 200
    fig = go.Figure(go.Candlestick(x=t.iloc[-n:], open=df["open"].iloc[-n:], high=df["high"].iloc[-n:],
                                   low=df["low"].iloc[-n:], close=df["close"].iloc[-n:], name="Price"))
    for length, color in [(20, "#06b6d4"), (50, "#f59e0b"), (200, "#9ca3af")]:
        fig.add_scatter(x=t.iloc[-n:], y=g.ema(c, length)[-n:], mode="lines", name=f"EMA{length}",
                        line=dict(color=color, width=1.4))
    levels = [("Entry", row["entry"], "#3b82f6"), ("SL", row["sl"], "#dc2626"), ("TP1", row["tp1"], "#22c55e"),
              ("TP2", row["tp2"], "#16a34a"), ("TP3", row["tp3"], "#0d9488"),
              (f"CHoCH {row['choch']}", row["choch_lvl"], "#a855f7")]
    for name, y, color in levels:
        if pd.notna(y):
            fig.add_hline(y=float(y), line_dash="dot", line_color=color, annotation_text=name,
                          annotation_position="right")
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=30, b=8),
                      legend=dict(orientation="h", y=1.04), title=f"{row['symbol']} | {tf}m")
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 GPR Strict Screener")
    st.caption("Bybit USDT Perpetual | port Pine V2.12 STRICT")
    with st.form("scan_form"):
        tf_label = st.selectbox("Timeframe", list(TF_LABELS), index=1)
        top = st.slider("Top N simbol (turnover 24h)", 20, 300, 120, step=10)
        min_turn = st.number_input("Min turnover 24h (juta USDT)", 0.0, 1000.0, 5.0, step=1.0)
        symbols_txt = st.text_input("Simbol tertentu (opsional)", placeholder="BTCUSDT, ETHUSDT")
        with st.expander("Parameter strategi"):
            strict = st.slider("Minimum Signal Score", 50, 100, 80)
            use_mtf = st.checkbox("Require MTF alignment", True)
            gpr_on = st.checkbox("GPR advisory", True)
            tok_on = st.checkbox("Token engine", True)
            sl_atr = st.number_input("Stop loss (x ATR)", 0.5, 5.0, 1.5, 0.1)
            tp1 = st.number_input("TP1 (RR)", 0.5, 10.0, 1.0, 0.1)
            tp2 = st.number_input("TP2 (RR)", 1.0, 10.0, 2.0, 0.1)
            tp3 = st.number_input("TP3 (RR)", 1.5, 20.0, 3.0, 0.1)
        base = st.selectbox("API domain Bybit", ["https://api.bybit.com", "https://api.bytick.com"])
        workers = st.slider("Thread paralel", 2, 12, 8)
        submitted = st.form_submit_button("🔍 Scan sekarang", type="primary", width="stretch")

    st.subheader("Filter tampilan")
    max_age = st.slider("Sinyal baru: umur ≤ (bar)", 0, 20, 3)
    choch_age = st.slider("CHoCH baru: umur ≤ (bar)", 0, 30, 3)
    choch_dir = st.radio("Arah CHoCH", ["Semua", "Bull", "Bear"], horizontal=True)
    min_score = st.slider("Tab skor: minimal skor", 0, 100, 60)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if submitted:
    tf = TF_LABELS[tf_label]
    cfg = g.Cfg(strict_score=strict, use_mtf=use_mtf, gpr_enabled=gpr_on, token_enabled=tok_on,
                sl_atr=sl_atr, tp1_rr=tp1, tp2_rr=tp2, tp3_rr=tp3)
    syms = [s.strip().upper() for s in symbols_txt.replace("\n", ",").split(",") if s.strip()]
    try:
        st.session_state["scan"] = run_scan(tf, cfg, top, min_turn, syms, workers, base)
    except Exception as e:                                      # noqa: BLE001
        st.session_state.pop("scan", None)
        st.error(f"Gagal mengambil data dari Bybit: {e}")
        st.info("Jika errornya 403 / timeout: Bybit memblokir IP server di region tertentu (mis. AS). "
                "Coba ganti 'API domain', atau jalankan di laptop / VPS non-AS.")
        st.stop()

scan = st.session_state.get("scan")
if not scan or not scan["raw"]:
    st.title("📡 GPR Futures Intelligence | Bybit Perp Screener")
    st.info("Atur parameter di sidebar lalu tekan **Scan sekarang**.")
    st.stop()

tf = scan["tf"]
df = prepare(scan["raw"], max_age, choch_age, choch_dir)
tf_name = {v: k for k, v in TF_LABELS.items()}[tf]

st.title("📡 GPR Futures Intelligence | Bybit Perp Screener")
st.caption(f"TF {tf_name} · {len(df)} simbol terscan ({scan['fails']} di-skip) · {scan['time']} · "
           "analisa pada candle yang sudah close")
if int(tf) >= 120 and scan["cfg"].use_mtf:
    st.warning("TF 2H/4H: MTF 3M tidak tersedia di Bybit, syarat MTF 3/4 menjadi 3 dari 3 TF.")

m = st.columns(6)
m[0].metric("LONG baru", int(((df["signal"] == "LONG") & df["is_fresh"]).sum()))
m[1].metric("SHORT baru", int(((df["signal"] == "SHORT") & df["is_fresh"]).sum()))
m[2].metric("Trade aktif", int(df["state"].isin(ACTIVE).sum()))
m[3].metric("CHoCH Bull", int((df["is_choch"] & (df["choch"] == "BULL")).sum()))
m[4].metric("CHoCH Bear", int((df["is_choch"] & (df["choch"] == "BEAR")).sum()))
m[5].metric("Pump", int((df["pump"] != "-").sum()))

tab_sig, tab_ch, tab_score, tab_chart = st.tabs(["🎯 Sinyal & Trade", "🔄 CHoCH", "📊 Skor", "📈 Chart"])

with tab_sig:
    d = df[df["is_sig"]].sort_values(["age", "best"], ascending=[True, False])
    if d.empty:
        st.info("Belum ada sinyal STRICT baru / trade aktif pada scan ini.")
    else:
        show_table(view_signals(d), ["Signal", "Trend", "GPR"])

with tab_ch:
    d = df[df["is_choch"]].sort_values(["choch_age", "best"], ascending=[True, False])
    if d.empty:
        st.info("Tidak ada CHoCH baru sesuai filter.")
    else:
        st.caption("CHoCH Bull = BOS naik saat bias sebelumnya bearish (arah long). "
                   "CHoCH Bear = BOS turun saat bias sebelumnya bullish (arah short). Ini filter struktur, bukan sinyal entry.")
        show_table(view_choch(d), ["CHoCH", "Trend", "Str", "GPR"])

with tab_score:
    d = df[df["best"] >= min_score].sort_values("best", ascending=False)
    if d.empty:
        st.info("Tidak ada simbol dengan skor sebesar itu.")
    else:
        show_table(view_scores(d), ["Bias", "Trend", "Str", "GPR"])

with tab_chart:
    options = df.sort_values(["is_sig", "is_choch", "best"], ascending=False)["symbol"].tolist()
    sym = st.selectbox("Simbol", options)
    row = df[df["symbol"] == sym].iloc[0]
    try:
        fig = make_chart(row, tf, scan["base"])
    except Exception as e:                                      # noqa: BLE001
        fig = None
        st.error(f"Gagal memuat chart: {e}")
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

export = df.drop(columns=["is_sig", "is_fresh", "is_choch"]).to_csv(index=False).encode()
st.sidebar.download_button("⬇️ Download CSV", export, file_name=f"gpr_screener_{tf_name}.csv", mime="text/csv",
                           width="stretch")
