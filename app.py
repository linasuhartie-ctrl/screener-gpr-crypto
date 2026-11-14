#!/usr/bin/env python3
"""
GPR FUTURES INTELLIGENCE PRO V2.12 STRICT  ->  Bybit USDT Perpetual Screener
==============================================================================
Port dari Pine Script v6 "GPR FUTURES INTELLIGENCE PRO V2.12 STRICT" ke Python.
Data: Bybit v5 public API (tanpa API key).

Install : pip install requests pandas numpy
Contoh  : python gpr_bybit_screener.py --tf 15 --top 150
          python gpr_bybit_screener.py --tf 60 --signals-only --csv hasil.csv
          python gpr_bybit_screener.py --tf 15 --choch-only      # screener CHoCH saja
          python gpr_bybit_screener.py --tf 60 --choch-only --choch-dir bull --choch-age 2
          python gpr_bybit_screener.py --tf 15 --watch          # scan tiap candle close

Telegram (opsional, hanya dengan --watch):
          export TG_TOKEN="xxx" ; export TG_CHAT_ID="123456"
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

BASE = "https://api.bybit.com"


# ----------------------------------------------------------------------------
# CONFIG  (nama & default sama dengan input Pine)
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    gpr_enabled: bool = True
    gpr_train: int = 40
    gpr_horizon: int = 12
    gpr_kernel: float = 1.0
    gpr_noise: float = 0.15
    token_enabled: bool = True
    token_len: int = 7
    pump_enabled: bool = True
    pump_vol_mult: float = 2.0
    pump_move_pct: float = 3.0
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    vol_len: int = 20
    strict_score: float = 80
    use_mtf: bool = True
    pivot_len: int = 5
    structure_mem: int = 10
    pullback_mem: int = 5
    max_div_gap: int = 30
    sl_atr: float = 1.5
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp3_rr: float = 3.0


TF_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
              "60": 3600, "120": 7200, "240": 14400}


def mtf_set(tf: str):
    """Mapping MTF persis seperti Pine (tf1..tf4). None = tidak tersedia di Bybit (3M/12M)."""
    s = TF_SECONDS[tf]
    if s <= 300:
        return ["15", "60", "240", "D"]
    if s <= 900:
        return ["60", "240", "D", "W"]
    if s <= 3600:
        return ["240", "D", "W", "M"]
    return ["D", "W", "M", None]          # 2H / 4H: tf4 = 3M tidak ada di Bybit


# ----------------------------------------------------------------------------
# INDIKATOR (setara ta.* di Pine)
# ----------------------------------------------------------------------------
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rma(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0 or len(x) - valid[0] < n:
        return out
    s = valid[0] + n - 1
    out[s] = x[valid[0]:s + 1].mean()
    a = 1.0 / n
    for i in range(s + 1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi_(close, n):
    ch = np.diff(close, prepend=np.nan)
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0))
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + ru / rd)
    out = np.where(rd == 0, 100.0, out)
    out = np.where(np.isnan(ru) | np.isnan(rd), np.nan, out)
    return out


def atr_(high, low, close, n):
    pc = np.roll(close, 1)
    tr = np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])
    tr[0] = high[0] - low[0]
    return rma(tr, n)


def pivots(x, L, is_high=True):
    """ta.pivothigh / ta.pivotlow: nilai muncul di bar i untuk pivot di bar i-L."""
    s = pd.Series(x)
    roll = s.rolling(2 * L + 1).max() if is_high else s.rolling(2 * L + 1).min()
    center = s.shift(L)
    return center.where(center == roll).to_numpy()


def barssince(b):
    out = np.full(len(b), np.nan)
    last = -1
    for i, v in enumerate(b):
        if v:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def shift(a, k=1):
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


# ----------------------------------------------------------------------------
# MTF  (bull[1]/bear[1] dari EMA20/EMA50 HTF, lookahead_on -> bar HTF terakhir yg sudah close)
# ----------------------------------------------------------------------------
def mtf_flags(ts_main: np.ndarray, htf: pd.DataFrame | None):
    n = len(ts_main)
    if htf is None or len(htf) < 3:
        return np.zeros(n, bool), np.zeros(n, bool)
    c = htf["close"].to_numpy()
    e20, e50 = ema(c, 20), ema(c, 50)
    bull = (c > e20) & (e20 > e50)
    bear = (c < e20) & (e20 < e50)
    h = pd.DataFrame({
        "ts": htf["ts"].to_numpy(),
        "bull": np.r_[False, bull[:-1]],
        "bear": np.r_[False, bear[:-1]],
    })
    left = pd.DataFrame({"ts": ts_main})
    m = pd.merge_asof(left, h, on="ts", direction="backward")
    return m["bull"].fillna(False).to_numpy(bool), m["bear"].fillna(False).to_numpy(bool)


# ----------------------------------------------------------------------------
# GPR ADVISORY (Matern 3/2, ATR-normalized) - vektor untuk semua bar
# ----------------------------------------------------------------------------
def gpr_series(close, atr, cfg: Cfg):
    n = len(close)
    T = cfg.gpr_train
    vs = np.maximum(atr / close, 1e-6)
    lr = np.full(n, np.nan)
    lr[1:] = np.log(close[1:] / close[:-1])
    feat = lr / vs                              # feature bar t  : log(c[t]/c[t-1]) / vs[t]
    tgt = np.full(n, np.nan)
    tgt[:-1] = lr[1:] / vs[:-1]                 # target  bar t  : log(c[t+1]/c[t]) / vs[t]

    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    fc = np.full(n, np.nan)
    pup = np.full(n, 0.5)
    if not cfg.gpr_enabled or n <= T + 2:
        return mu, sigma, fc, pup

    Wf = sliding_window_view(feat, T)[: n - T]  # baris k -> feat[k : k+T]  (t = k+T)
    Wt = sliding_window_view(tgt, T)[: n - T]
    cur = feat[T:]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dist = np.abs(cur[:, None] - Wf) / cfg.gpr_kernel
        rt = math.sqrt(3.0) * dist
        w = (1.0 + rt) * np.exp(-rt)
        ws = w.sum(1)
        wsum = (w * Wt).sum(1)
        wsq = (w * Wt * Wt).sum(1)
        m = np.where(ws > 0, wsum / ws, np.nan)
        var = np.where(ws > 0, np.maximum(wsq / ws - m * m, 0.0), np.nan)
        sg = np.sqrt(var + cfg.gpr_noise ** 2)
        f = (np.exp(m * vs[T:] * cfg.gpr_horizon) - 1.0) * 100
        z = np.where(sg > 0, np.clip(m / sg, -10, 10), 0.0)
        p = 1.0 / (1.0 + np.exp(-2.0 * z))

    idx = np.arange(T, n)
    ok = idx > (T + cfg.atr_len + 5)
    ok &= np.isfinite(m) & np.isfinite(sg)
    mu[idx[ok]] = m[ok]
    sigma[idx[ok]] = sg[ok]
    fc[idx[ok]] = f[ok]
    pup[idx[ok]] = p[ok]
    return mu, sigma, fc, pup


# ----------------------------------------------------------------------------
# ANALISA 1 SIMBOL  (semua bar -> state bar terakhir yang sudah close)
# ----------------------------------------------------------------------------
def analyze(df: pd.DataFrame, htfs: list, tick: float, cfg: Cfg):
    """df: OHLCV (ascending, hanya bar CLOSED). htfs: list 4 DataFrame HTF / None."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    ts = df["ts"].to_numpy()
    n = len(c)

    # ---- 02. core
    ef, em, es = ema(c, cfg.ema_fast), ema(c, cfg.ema_mid), ema(c, cfg.ema_slow)
    rsi = rsi_(c, cfg.rsi_len)
    atr = atr_(h, l, c, cfg.atr_len)
    volma = pd.Series(v).rolling(cfg.vol_len).mean().to_numpy()
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)

    trend_bull = (c > ef) & (ef > em) & (em > es)
    trend_bear = (c < ef) & (ef < em) & (em < es)
    mom_bull = (rsi > 50) & (hist > 0)
    mom_bear = (rsi < 50) & (hist < 0)
    vol_bull = (v > volma) & (c > o)
    vol_bear = (v > volma) & (c < o)
    vdp = v * (c - o) / np.maximum(h - l, tick)

    # ---- 03. MTF
    bulls, bears = zip(*[mtf_flags(ts, x) for x in htfs])
    bull_cnt = sum(b.astype(int) for b in bulls)
    bear_cnt = sum(b.astype(int) for b in bears)
    mtf_bull = bull_cnt >= 3
    mtf_bear = bear_cnt >= 3
    long_mtf_ok = np.ones(n, bool) if not cfg.use_mtf else mtf_bull
    short_mtf_ok = np.ones(n, bool) if not cfg.use_mtf else mtf_bear

    # ---- 04. structure
    L = cfg.pivot_len
    ph, pl = pivots(h, L, True), pivots(l, L, False)
    lsh = np.full(n, np.nan)
    lsl = np.full(n, np.nan)
    ch_, cl_ = np.nan, np.nan
    for i in range(n):
        if not np.isnan(ph[i]):
            ch_ = ph[i]
        if not np.isnan(pl[i]):
            cl_ = pl[i]
        lsh[i], lsl[i] = ch_, cl_
    with np.errstate(invalid="ignore"):
        cross_up = (c > lsh) & (shift(c) <= shift(lsh))
        cross_dn = (c < lsl) & (shift(c) >= shift(lsl))
        bull_bos = ~np.isnan(lsh) & cross_up
        bear_bos = ~np.isnan(lsl) & cross_dn
        bsl_sweep = ~np.isnan(lsh) & (h > lsh) & (c < lsh)
        ssl_sweep = ~np.isnan(lsl) & (l < lsl) & (c > lsl)
    bias = np.zeros(n, int)
    bull_choch = np.zeros(n, bool)
    bear_choch = np.zeros(n, bool)
    b = 0
    for i in range(n):
        bull_choch[i] = bool(bull_bos[i]) and b == -1
        bear_choch[i] = bool(bear_bos[i]) and b == 1
        if bull_bos[i]:
            b = 1
        if bear_bos[i]:
            b = -1
        bias[i] = b

    bs_bull = barssince(bull_bos)          # bullStructureEvent == bullBOS (CHoCH adalah subset)
    bs_bear = barssince(bear_bos)
    bs_ssl = barssince(ssl_sweep)
    bs_bsl = barssince(bsl_sweep)
    with np.errstate(invalid="ignore"):
        rec_bull_struct = bs_bull <= cfg.structure_mem
        rec_bear_struct = bs_bear <= cfg.structure_mem
        rec_ssl = bs_ssl <= cfg.structure_mem
        rec_bsl = bs_bsl <= cfg.structure_mem

    # ---- 05. RSI divergence
    rph, rpl = pivots(rsi, L, True), pivots(rsi, L, False)
    bear_div = np.zeros(n, bool)
    bull_div = np.zeros(n, bool)
    pph = prh = pl_ = prl = np.nan
    phb = plb = -10 ** 9
    for i in range(n):
        if not np.isnan(ph[i]) and not np.isnan(rph[i]):
            cb = i - L
            bear_div[i] = (not np.isnan(pph)) and cb - phb <= cfg.max_div_gap and ph[i] > pph and rph[i] < prh
            pph, prh, phb = ph[i], rph[i], cb
        if not np.isnan(pl[i]) and not np.isnan(rpl[i]):
            cb = i - L
            bull_div[i] = (not np.isnan(pl_)) and cb - plb <= cfg.max_div_gap and pl[i] < pl_ and rpl[i] > prl
            pl_, prl, plb = pl[i], rpl[i], cb
    with np.errstate(invalid="ignore"):
        rec_bull_div = barssince(bull_div) <= cfg.max_div_gap
        rec_bear_div = barssince(bear_div) <= cfg.max_div_gap

    # ---- 06. pullback sequence
    with np.errstate(invalid="ignore"):
        long_pb = (l <= ef) & (c >= ef * 0.995) & (v < volma)
        short_pb = (h >= ef) & (c <= ef * 1.005) & (v < volma)
        bs_lpb, bs_spb = barssince(long_pb), barssince(short_pb)
        rec_lpb = bs_lpb <= cfg.pullback_mem
        rec_spb = bs_spb <= cfg.pullback_mem
        long_conf = (c > o) & (c > ef) & vol_bull
        short_conf = (c < o) & (c < ef) & vol_bear
        long_seq = rec_bull_struct & rec_lpb & (bs_bull > bs_lpb) & (bs_lpb >= 1)
        short_seq = rec_bear_struct & rec_spb & (bs_bear > bs_spb) & (bs_spb >= 1)

    # ---- 07. pump
    pchg = np.zeros(n)
    pchg[1:] = np.where(c[:-1] != 0, (c[1:] - c[:-1]) / c[:-1] * 100, 0.0)
    with np.errstate(invalid="ignore"):
        pump_vol = v > volma * cfg.pump_vol_mult
    pump_up = cfg.pump_enabled & (pchg >= cfg.pump_move_pct) & pump_vol
    pump_dn = cfg.pump_enabled & (pchg <= -cfg.pump_move_pct) & pump_vol

    # ---- 08. token
    trsi = rsi_(c, cfg.token_len)
    with np.errstate(invalid="ignore"):
        tok_bull = cfg.token_enabled & (trsi > 60) & (c > ef) & (vdp > 0)
        tok_bear = cfg.token_enabled & (trsi < 40) & (c < ef) & (vdp < 0)
    tok_score = np.where(tok_bull, 100, np.where(tok_bear, 0, 50))

    # ---- 09. GPR
    _, _, gfc, gpup = gpr_series(c, atr, cfg)
    g_bull = gpup >= 0.60
    g_bear = gpup <= 0.40

    # ---- 10. score 0-100 (dinamis)
    lr_ = np.zeros(n)
    sr_ = np.zeros(n)
    mx = 0.0
    for lw, sw, w in [(trend_bull, trend_bear, 20), (mom_bull, mom_bear, 10), (rec_lpb, rec_spb, 10),
                      (long_conf, short_conf, 10), (long_seq, short_seq, 10)]:
        lr_ += np.where(lw, w, 0)
        sr_ += np.where(sw, w, 0)
        mx += w
    if cfg.use_mtf:
        lr_ += np.where(mtf_bull, 15, 0)
        sr_ += np.where(mtf_bear, 15, 0)
        mx += 15
    if cfg.gpr_enabled:
        lr_ += np.where(g_bull, 10, 0)
        sr_ += np.where(g_bear, 10, 0)
        mx += 10
    lr_ += np.where(rec_ssl, 5, 0)
    sr_ += np.where(rec_bsl, 5, 0)
    mx += 5
    if cfg.token_enabled:
        lr_ += np.where(tok_bull, 5, 0)
        sr_ += np.where(tok_bear, 5, 0)
        mx += 5
    lr_ += np.where(rec_bull_div, 5, 0)
    sr_ += np.where(rec_bear_div, 5, 0)
    mx += 5
    long_score, short_score = lr_ / mx * 100, sr_ / mx * 100

    with np.errstate(invalid="ignore"):
        long_setup = (long_score >= cfg.strict_score) & trend_bull & mom_bull & long_seq & long_conf & long_mtf_ok
        short_setup = (short_score >= cfg.strict_score) & trend_bear & mom_bear & short_seq & short_conf & short_mtf_ok

    # ---- 11. trade level management (state machine bar-by-bar, konservatif: SL dulu)
    tp1 = cfg.tp1_rr
    tp2 = max(cfg.tp2_rr, tp1 + 0.1)
    tp3 = max(cfg.tp3_rr, tp2 + 0.1)
    d = stage = 0
    entry = sl = t1 = t2 = t3 = np.nan
    sig = None                       # info sinyal terakhir
    last_event, last_event_bar = "", -1
    for i in range(n):
        closed = False
        ev = ""
        if d == 1:
            if l[i] <= sl:
                ev, closed, d, stage = "SL HIT" if stage == 0 else "TRAIL SL", True, 0, 0
            elif h[i] >= t3:
                ev, closed, d, stage = "TP3", True, 0, 0
            elif h[i] >= t2:
                ev = "TP2" if stage < 2 else ""
                stage, sl = 2, t1
            elif h[i] >= t1 and stage == 0:
                ev, stage, sl = "TP1/BE", 1, entry
        elif d == -1:
            if h[i] >= sl:
                ev, closed, d, stage = "SL HIT" if stage == 0 else "TRAIL SL", True, 0, 0
            elif l[i] <= t3:
                ev, closed, d, stage = "TP3", True, 0, 0
            elif l[i] <= t2:
                ev = "TP2" if stage < 2 else ""
                stage, sl = 2, t1
            elif l[i] <= t1 and stage == 0:
                ev, stage, sl = "TP1/BE", 1, entry
        if ev:
            last_event, last_event_bar = ev, i

        if d == 0 and not closed:
            if long_setup[i] or short_setup[i]:
                sgn = 1 if long_setup[i] else -1
                d, stage, entry = sgn, 0, c[i]
                sl = c[i] - sgn * atr[i] * cfg.sl_atr
                risk = abs(entry - sl)
                t1, t2, t3 = (entry + sgn * risk * r for r in (tp1, tp2, tp3))
                sig = dict(dir=sgn, bar=i, entry=entry, sl0=sl, t1=t1, t2=t2, t3=t3)
                last_event, last_event_bar = "ENTRY", i

    i = n - 1
    cands = [(a, k) for a, k in ((barssince(bull_choch)[-1], "BULL"), (barssince(bear_choch)[-1], "BEAR"))
             if not np.isnan(a)]
    ch_age, ch_dir = min(cands) if cands else (np.nan, "-")
    ch_lvl = np.nan
    if cands:
        j = i - int(ch_age)
        ch_lvl = lsh[j] if ch_dir == "BULL" else lsl[j]      # swing yang ditembus
    out = dict(
        price=c[i],
        trend="BULL" if trend_bull[i] else "BEAR" if trend_bear[i] else "SIDE",
        struct="BULL" if bias[i] == 1 else "BEAR" if bias[i] == -1 else "-",
        mtf=f"{bull_cnt[i]}/{bear_cnt[i]}",
        rsi=rsi[i],
        token=int(tok_score[i]),
        pump="UP" if pump_up[i] else "DOWN" if pump_dn[i] else "-",
        gpr="BULL" if g_bull[i] else "BEAR" if g_bear[i] else "NEUT",
        gpr_fc=gfc[i],
        l_score=long_score[i],
        s_score=short_score[i],
        seq="L-READY" if long_seq[i] else "S-READY" if short_seq[i] else "-",
        liq="BSL" if rec_bsl[i] else "SSL" if rec_ssl[i] else "-",
        choch=ch_dir, choch_age=ch_age, choch_lvl=ch_lvl,
    )

    out.update(signal="-", age=np.nan, state="-", entry=np.nan, sl=np.nan, tp1=np.nan, tp2=np.nan, tp3=np.nan)
    if sig is not None:
        out["signal"] = "LONG" if sig["dir"] == 1 else "SHORT"
        out["age"] = i - sig["bar"]
        out["entry"], out["tp1"], out["tp2"], out["tp3"] = sig["entry"], sig["t1"], sig["t2"], sig["t3"]
        if d != 0:
            out["state"] = ["INIT SL", "BE LOCKED", "LVL1 LOCKED"][stage]
            out["sl"] = sl                        # trailing SL aktif
        else:
            out["state"] = "CLOSED:" + last_event
            out["sl"] = sig["sl0"]
        out["sig_ts"] = int(ts[sig["bar"]])
    return out


# ----------------------------------------------------------------------------
# BYBIT API
# ----------------------------------------------------------------------------
def api_get(session, path, params, retries=5):
    err = None
    for k in range(retries):
        try:
            r = session.get(BASE + path, params=params, timeout=12)
            if r.status_code in (403, 429):
                time.sleep(2 + 2 * k)
                continue
            j = r.json()
            if j.get("retCode") == 0:
                return j["result"]
            if j.get("retCode") in (10006, 10018):      # rate limit
                time.sleep(1 + k)
                continue
            err = RuntimeError(f"{path}: {j.get('retCode')} {j.get('retMsg')}")
            break
        except (requests.RequestException, ValueError) as e:
            err = e
            time.sleep(0.5 + k)
    raise err or RuntimeError("request failed")


def klines(session, symbol, interval, limit):
    res = api_get(session, "/v5/market/kline",
                  {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit})
    rows = res["list"]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df.sort_values("ts").reset_index(drop=True)


def universe(session, min_turnover, top, explicit=None):
    inst, cursor = {}, None
    while True:
        p = {"category": "linear", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        res = api_get(session, "/v5/market/instruments-info", p)
        for i in res["list"]:
            if i.get("contractType") == "LinearPerpetual" and i.get("status") == "Trading" \
                    and i.get("quoteCoin") == "USDT":
                inst[i["symbol"]] = float(i["priceFilter"]["tickSize"])
        cursor = res.get("nextPageCursor")
        if not cursor:
            break
    tick = api_get(session, "/v5/market/tickers", {"category": "linear"})["list"]
    rows = []
    for t in tick:
        s = t["symbol"]
        if s not in inst:
            continue
        rows.append(dict(symbol=s, tick=inst[s], chg24=float(t["price24hPcnt"] or 0) * 100,
                         turn=float(t["turnover24h"] or 0), fund=float(t["fundingRate"] or 0) * 100))
    u = pd.DataFrame(rows)
    if explicit:
        u = u[u["symbol"].isin([s.upper() for s in explicit])]
    else:
        u = u[u["turn"] >= min_turnover].sort_values("turn", ascending=False).head(top)
    return u.reset_index(drop=True)


def scan_symbol(session, row, tf, cfg, bars, min_bars):
    sym = row["symbol"]
    main = klines(session, sym, tf, bars)
    if main is None or len(main) < min_bars + 1:
        return None
    main = main.iloc[:-1].reset_index(drop=True)          # buang bar yang masih berjalan (barstate.isconfirmed)
    htfs = []
    for itv in mtf_set(tf):
        htfs.append(klines(session, sym, itv, 250) if (itv and cfg.use_mtf) else None)
    r = analyze(main, htfs, row["tick"], cfg)
    r.update(symbol=sym, chg24=row["chg24"], turn=row["turn"] / 1e6, fund=row["fund"])
    return r


# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
def fmt_price(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6g}"


def render(results, args):
    if not results:
        print("Tidak ada data.")
        return None
    df = pd.DataFrame(results)
    df["best"] = df[["l_score", "s_score"]].max(axis=1)
    df["bias"] = np.where(df["l_score"] >= df["s_score"], "L", "S")
    fresh = df["signal"].ne("-") & (df["age"] <= args.max_age)
    active = df["state"].isin(["INIT SL", "BE LOCKED", "LVL1 LOCKED"])
    ch_any = df["choch"].ne("-") & (df["choch_age"] <= args.choch_age)       # CHoCH baru (tampil)
    ch_sel = ch_any.copy()                                                    # CHoCH sesuai filter arah
    if args.choch_dir != "both":
        ch_sel &= df["choch"].eq(args.choch_dir.upper())
    keep = fresh | active | ch_sel | (df["best"] >= args.min_score)
    if args.signals_only:
        keep = fresh | active
    if args.choch_only:
        keep = ch_sel
    for col, empty in [("choch", "-"), ("choch_age", np.nan), ("choch_lvl", np.nan)]:
        df.loc[~ch_any, col] = empty
    # sinyal lama yang sudah closed tidak ditampilkan (kolom dikosongkan)
    stale = df["signal"].ne("-") & ~fresh & ~active
    for col, empty in [("signal", "-"), ("state", "-"), ("age", np.nan), ("entry", np.nan),
                       ("sl", np.nan), ("tp1", np.nan), ("tp2", np.nan), ("tp3", np.nan)]:
        df.loc[stale, col] = empty
    df = df[keep].copy()
    df["_fresh"] = fresh[keep]
    if args.choch_only:
        df = df.sort_values(["choch_age", "best"], ascending=[True, False]).reset_index(drop=True)
    else:
        df = df.sort_values(["_fresh", "age", "best"], ascending=[False, True, False],
                            na_position="last").reset_index(drop=True)

    view = pd.DataFrame({
        "Symbol": df["symbol"],
        "Price": df["price"].map(fmt_price),
        "24h%": df["chg24"].map("{:+.1f}".format),
        "Vol$M": df["turn"].map("{:.0f}".format),
        "Trend": df["trend"], "Str": df["struct"],
        "CHoCH": df["choch"] + df["choch_age"].map(lambda a: "" if np.isnan(a) else f" -{int(a)}b"),
        "CH Lvl": df["choch_lvl"].map(fmt_price), "Liq": df["liq"],
        "MTF b/s": df["mtf"],
        "RSI": df["rsi"].map("{:.0f}".format),
        "Tok": df["token"], "Pump": df["pump"],
        "GPR": df["gpr"], "GPR%": df["gpr_fc"].map(lambda x: "" if np.isnan(x) else f"{x:+.2f}"),
        "L/S": df["l_score"].map("{:.0f}".format) + "/" + df["s_score"].map("{:.0f}".format),
        "Seq": df["seq"],
        "Signal": df["signal"] + df["age"].map(lambda a: "" if np.isnan(a) else f" -{int(a)}b"),
        "State": df["state"],
        "Entry": df["entry"].map(fmt_price), "SL": df["sl"].map(fmt_price),
        "TP1": df["tp1"].map(fmt_price), "TP2": df["tp2"].map(fmt_price), "TP3": df["tp3"].map(fmt_price),
        "Fund%": df["fund"].map("{:+.3f}".format),
    })
    print(f"\n=== GPR STRICT SCREENER | Bybit USDT Perp | TF {args.tf} | "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} | scanned {len(results)} | tampil {len(view)} ===")
    print(view.to_string(index=False) if len(view) else "(tidak ada yang memenuhi filter)")
    return df


def tg_send(text):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not tok or not chat:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=10)
    except requests.RequestException:
        pass


def run_once(args, cfg, alerted):
    s = requests.Session()
    s.headers["User-Agent"] = "gpr-screener/1.0"
    uni = universe(s, args.min_turnover, args.top, args.symbols)
    print(f"Universe: {len(uni)} perpetual USDT. Mengambil data...")
    results, fails = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_symbol, s, r, args.tf, cfg, args.bars, max(cfg.ema_slow + 50, 260))
                for r in uni.to_dict("records")]
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                if fails <= 3:
                    print("  skip:", e, file=sys.stderr)
    print(f"Selesai {time.time() - t0:.1f}s  (di-skip/gagal: {len(uni) - len(results)})")
    shown = render(results, args)
    if shown is not None and args.csv:
        shown.drop(columns=["_fresh"]).to_csv(args.csv, index=False)
        print("CSV:", args.csv)
    if args.watch and shown is not None:
        alert_choch(shown, args, alerted)
        for r in shown.to_dict("records"):
            if r["signal"] != "-" and r["age"] == 0:
                key = (r["symbol"], args.tf, r.get("sig_ts"))
                if key not in alerted:
                    alerted.add(key)
                    tg_send(f"GPR STRICT {r['signal']} {r['symbol']} [{args.tf}]\n"
                            f"Entry {fmt_price(r['entry'])} | SL {fmt_price(r['sl'])}\n"
                            f"TP1 {fmt_price(r['tp1'])} | TP2 {fmt_price(r['tp2'])} | TP3 {fmt_price(r['tp3'])}\n"
                            f"Score L/S {r['l_score']:.0f}/{r['s_score']:.0f}")


def alert_choch(shown, args, alerted):
    for r in shown.to_dict("records"):
        if r["choch"] != "-" and r["choch_age"] == 0:
            key = ("CH", r["symbol"], args.tf, r["choch"], round(float(r["price"]), 8))
            if key not in alerted:
                alerted.add(key)
                tg_send(f"CHoCH {r['choch']} {r['symbol']} [{args.tf}]\n"
                        f"Level {fmt_price(r['choch_lvl'])} | Price {fmt_price(r['price'])}\n"
                        f"Liq {r['liq']} | Trend {r['trend']} | MTF {r['mtf']} | Score L/S {r['l_score']:.0f}/{r['s_score']:.0f}")


def main():
    p = argparse.ArgumentParser(description="GPR Futures Intelligence STRICT - Bybit perpetual screener")
    p.add_argument("--tf", default="15", choices=list(TF_SECONDS), help="timeframe menit (default 15)")
    p.add_argument("--top", type=int, default=150, help="jumlah simbol teratas berdasar turnover 24h")
    p.add_argument("--min-turnover", type=float, default=5e6, help="min turnover 24h (USDT)")
    p.add_argument("--symbols", nargs="*", help="scan simbol tertentu saja, mis. BTCUSDT ETHUSDT")
    p.add_argument("--min-score", type=float, default=60, help="tampilkan skor L/S >= nilai ini")
    p.add_argument("--max-age", type=int, default=3, help="sinyal dianggap baru jika umur <= N bar")
    p.add_argument("--signals-only", action="store_true", help="hanya sinyal baru / trade yang masih aktif")
    p.add_argument("--bars", type=int, default=1000, help="jumlah candle utama (maks 1000)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--csv", help="simpan hasil ke file CSV")
    p.add_argument("--watch", action="store_true", help="scan ulang setiap candle close (+Telegram jika di-set)")
    p.add_argument("--choch-only", action="store_true", help="screener CHoCH: hanya simbol dengan CHoCH baru")
    p.add_argument("--choch-age", type=int, default=3, help="CHoCH dianggap baru jika umur <= N bar (default 3)")
    p.add_argument("--choch-dir", choices=["both", "bull", "bear"], default="both", help="filter arah CHoCH")
    # parameter strategi
    p.add_argument("--strict", type=float, default=80, help="Minimum Signal Score")
    p.add_argument("--no-mtf", action="store_true")
    p.add_argument("--no-gpr", action="store_true")
    p.add_argument("--no-token", action="store_true")
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--tp1", type=float, default=1.0)
    p.add_argument("--tp2", type=float, default=2.0)
    p.add_argument("--tp3", type=float, default=3.0)
    args = p.parse_args()

    cfg = Cfg(strict_score=args.strict, use_mtf=not args.no_mtf, gpr_enabled=not args.no_gpr,
              token_enabled=not args.no_token, sl_atr=args.sl_atr,
              tp1_rr=args.tp1, tp2_rr=args.tp2, tp3_rr=args.tp3)
    if TF_SECONDS[args.tf] >= 7200 and cfg.use_mtf:
        print("Catatan: pada TF 2H/4H, MTF 3M tidak tersedia di Bybit -> syarat MTF 3/4 menjadi 3 dari 3 TF.")

    alerted = set()
    while True:
        run_once(args, cfg, alerted)
        if not args.watch:
            break
        sec = TF_SECONDS[args.tf]
        now = time.time()
        wait = (now // sec + 1) * sec + 8 - now
        print(f"Menunggu candle close berikutnya ({wait:.0f}s)...")
        time.sleep(wait)


if __name__ == "__main__":
    main()
