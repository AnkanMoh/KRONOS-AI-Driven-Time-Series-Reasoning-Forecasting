import json, pickle, numpy as np, pandas as pd, torch
from util import load_cfg, ensure_dir, set_seed, load_or_build_features, KronosP100, device_from_cfg

def _consistency(o,c,l,h):
    lo = float(min(o,c,l,h)); hi = float(max(o,c,l,h))
    o = float(min(max(o, lo), hi)); c = float(min(max(c, lo), hi))
    return o, c, lo, hi

def main():
    cfg = load_cfg("config.yaml"); set_seed(cfg.get("random_seed",123)); ensure_dir(cfg["artifacts_dir"])
    dev = device_from_cfg(cfg)
    ckpt = torch.load(cfg["model_path"], map_location="cpu"); feature_cols = ckpt["feature_cols"]
    model = KronosP100(len(feature_cols), cfg).to(dev); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with open(cfg["scaler_path"], "rb") as f: scalers = pickle.load(f)
    ctx = int(cfg["context_length"])
    v_start = pd.to_datetime(cfg["valid_start_date"]).date(); v_end = pd.to_datetime(cfg["valid_end_date"]).date()
    results = {}
    for tk in cfg["ticker_list"]:
        df_feat = load_or_build_features(tk, cfg, cfg["valid_end_date"])
        if df_feat is None or df_feat.empty or tk not in scalers: continue
        if not all(c in df_feat.columns for c in feature_cols): continue
        sx_mean = np.array(scalers[tk]["x_mean"], dtype=np.float32)
        sx_scale= np.where(np.array(scalers[tk]["x_scale"], dtype=np.float32)==0, 1.0, np.array(scalers[tk]["x_scale"], dtype=np.float32))
        py_mean = np.array(scalers[tk]["y_price_mean"], dtype=np.float32)
        py_scale= np.where(np.array(scalers[tk]["y_price_scale"], dtype=np.float32)==0, 1.0, np.array(scalers[tk]["y_price_scale"], dtype=np.float32))
        X = df_feat[feature_cols].values.astype(np.float32); dates = df_feat.index
        pred_map = {}
        for end_idx in range(ctx-1, len(df_feat)-1):
            d = dates[end_idx+1].date()
            if d < v_start or d > v_end: continue
            x_ctx = X[end_idx-(ctx-1):end_idx+1]
            x_norm = (x_ctx - sx_mean) / sx_scale
            with torch.no_grad():
                price_std, vol_log = model(torch.tensor(x_norm[None, ...], dtype=torch.float32, device=dev))
                price_std = price_std.cpu().numpy()[0]; vol_log = vol_log.cpu().numpy()[0,0]
            price = price_std * py_scale + py_mean
            o, c, l, h = float(price[0]), float(price[1]), float(price[2]), float(price[3])
            o, c, l, h = _consistency(o,c,l,h)
            vol = int(np.clip(np.expm1(vol_log), 0, 5e9))
            pred_map[str(d)] = {"Open": float(f"{o:.2f}"), "Close": float(f"{c:.2f}"), "Low": float(f"{l:.2f}"), "High": float(f"{h:.2f}"), "Volume": vol}
        if pred_map: results[tk] = pred_map
    with open(cfg["predictions_path"], "w") as f: json.dump(results, f, indent=2)
    print("OK")

if __name__ == "__main__":
    main()
