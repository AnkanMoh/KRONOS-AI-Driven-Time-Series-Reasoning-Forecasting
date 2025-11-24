import os, pickle, yaml
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset

TARGETS = ["Open","Close","Low","High","Volume"]

def load_cfg(path: str) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 123):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    end_exc = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_exc, progress=False, auto_adjust=False, group_by="column")
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    keep = ["Open","High","Low","Close","Volume"]
    for k in keep:
        if k not in df.columns: df[k] = np.nan
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    out = df.copy(); close = out["Close"].astype(float)
    for w in (5, 10, 20, 50): out[f"SMA{w}"] = SMAIndicator(close, window=w).sma_indicator()
    out["RSI14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    out["MACD"] = macd.macd(); out["MACD_Signal"] = macd.macd_signal(); out["MACD_Hist"] = macd.macd_diff()
    bb = BollingerBands(close, window=20, window_dev=2)
    out["BB_High"] = bb.bollinger_hband(); out["BB_Low"]  = bb.bollinger_lband()
    out["Ret1"] = close.pct_change(); out["LogRet1"] = np.log(close.replace(0, np.nan)).diff(); out["Vol20"] = out["LogRet1"].rolling(20).std()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def build_features(df: pd.DataFrame, use_indicators: bool) -> pd.DataFrame:
    return add_indicators(df) if use_indicators else df.copy()

def load_or_build_features(ticker: str, cfg: dict, full_end: str) -> pd.DataFrame:
    ensure_dir(cfg["cache_dir"])
    cache_path = os.path.join(cfg["cache_dir"], f"{ticker}.parquet")
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # quick freshness check
            if df.index.max().date() >= pd.to_datetime(cfg["valid_end_date"]).date():
                return df
        except Exception:
            pass
    raw = yf_download(ticker, cfg["train_start_date"], full_end)
    if raw.empty: return raw
    feat = build_features(raw, cfg.get("use_indicators", True))
    feat.to_parquet(cache_path)
    return feat

@dataclass
class TickerScalers:
    x: StandardScaler
    y_price: StandardScaler

def fit_scalers_per_ticker(train_df: pd.DataFrame, feature_cols: List[str]) -> TickerScalers:
    xs = train_df[feature_cols].values.astype(np.float32)
    prices = train_df[["Open","Close","Low","High"]].values.astype(np.float32)
    sx = StandardScaler().fit(xs); sy_price = StandardScaler().fit(prices)
    return TickerScalers(sx, sy_price)

class WindowDataset(Dataset):
    def __init__(self, x, prices, volumes, ctx, sx: StandardScaler, sy_price: StandardScaler, ticker_id: int):
        self.ctx = int(ctx)
        self.X = (x - sx.mean_) / np.where(sx.scale_==0, 1.0, sx.scale_)
        self.P = (prices - sy_price.mean_) / np.where(sy_price.scale_==0, 1.0, sy_price.scale_)
        self.V = np.log1p(volumes)
        self.T = x.shape[0]
        self.ticker_id = ticker_id
    def __len__(self): return max(0, self.T - self.ctx - 1)
    def __getitem__(self, idx):
        t = idx + self.ctx - 1
        x_ctx = self.X[t-(self.ctx-1):t+1]
        y_price_next = self.P[t+1]
        y_vol_next = self.V[t+1]
        return (torch.from_numpy(x_ctx).float(),
                torch.from_numpy(y_price_next).float(),
                torch.tensor(y_vol_next, dtype=torch.float32))

class Encoder(nn.Module):
    def __init__(self, f_in: int, d_model=160, n_heads=5, n_layers=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        z = self.proj(x)
        h = self.encoder(z)
        return self.norm(h[:, -1, :])

class Heads(nn.Module):
    def __init__(self, d_model=160, hidden=160, dropout=0.15):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout))
        self.price = nn.Linear(hidden, 4)
        self.vol   = nn.Linear(hidden, 1)
    def forward(self, h):
        s = self.shared(h)
        return self.price(s), self.vol(s)

class KronosP100(nn.Module):
    def __init__(self, f_in, cfg):
        super().__init__()
        self.enc = Encoder(f_in, cfg.get("d_model",160), cfg.get("n_heads",5), cfg.get("n_layers",3), cfg.get("dropout",0.15))
        self.heads = Heads(cfg.get("d_model",160), hidden=cfg.get("d_model",160), dropout=cfg.get("dropout",0.15))
    def forward(self, x):
        h = self.enc(x)
        return self.heads(h)

def device_from_cfg(cfg):
    if cfg.get("device") == "cpu": return torch.device("cpu")
    if cfg.get("device") == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
