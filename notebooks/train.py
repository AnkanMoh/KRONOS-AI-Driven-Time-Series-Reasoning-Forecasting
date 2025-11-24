import os, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from util import (load_cfg, ensure_dir, set_seed, build_features, load_or_build_features,
                  WindowDataset, fit_scalers_per_ticker, KronosP100, device_from_cfg)

def main():
    cfg = load_cfg("config.yaml")
    set_seed(cfg.get("random_seed",123))
    ensure_dir(cfg["artifacts_dir"]); ensure_dir(cfg["cache_dir"])
    dev = device_from_cfg(cfg)

    ctx = int(cfg["context_length"])
    gacc = int(cfg.get("grad_accum_steps", 1))
    use_ind = bool(cfg.get("use_indicators", True))

    datasets, scalers = [], {}
    feature_cols = None

    for i, tk in enumerate(cfg["ticker_list"]):
        df_feat = load_or_build_features(tk, cfg, cfg["valid_end_date"])
        if df_feat is None or df_feat.empty: continue
        if feature_cols is None: feature_cols = list(df_feat.columns)
        train_df = df_feat.loc[:pd.to_datetime(cfg["train_end_date"])]
        if len(train_df) < ctx + 2: continue
        ts = fit_scalers_per_ticker(train_df, feature_cols)
        scalers[tk] = {"x_mean": ts.x.mean_.tolist(), "x_scale": ts.x.scale_.tolist(),
                       "y_price_mean": ts.y_price.mean_.tolist(), "y_price_scale": ts.y_price.scale_.tolist(),
                       "feature_cols": feature_cols}
        X = train_df[feature_cols].values.astype(np.float32)
        P = train_df[["Open","Close","Low","High"]].values.astype(np.float32)
        V = train_df[["Volume"]].values.astype(np.float32).ravel()
        ds = WindowDataset(X, P, V, ctx=ctx, sx=ts.x, sy_price=ts.y_price, ticker_id=i)
        if len(ds) > 0: datasets.append(ds)

    if not datasets: raise RuntimeError("No training data after preprocessing.")

    train_ds = ConcatDataset(datasets)
    loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)

    f_in = len(feature_cols)
    model = KronosP100(f_in, cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay",0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,int(cfg["epochs"])))
    mse = nn.MSELoss(); ce  = nn.CrossEntropyLoss()
    lam = float(cfg.get("lambda_dir", 0.25))
    scaler = GradScaler(enabled=(dev.type=="cuda"))

    torch.backends.cudnn.benchmark = True

    def dir_logits(pred_close_std, last_close_std):
        d = pred_close_std - last_close_std
        return torch.stack([-d, -torch.abs(d), d], dim=1)

    epochs = int(cfg["epochs"])
    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False), start=1):
            xb, y_price_next, y_vol_next = batch
            xb = xb.to(dev); y_price_next = y_price_next.to(dev); y_vol_next = y_vol_next.to(dev)

            with autocast(enabled=(dev.type=="cuda")):
                price_std, vol_log = model(xb)
                vol_log = vol_log.squeeze(1)
                loss_price = mse(price_std, y_price_next)
                loss_vol   = mse(vol_log, y_vol_next)
                last_close_std = xb[:, -1, feature_cols.index("Close")]
                close_pred_std = price_std[:, 1]
                true_diff = y_price_next[:,1] - last_close_std
                y_dir = torch.where(true_diff>0, 2, torch.where(true_diff<0, 0, 1))
                logits = dir_logits(close_pred_std, last_close_std)
                loss_dir = ce(logits, y_dir.long())
                loss = loss_price + loss_vol + lam * loss_dir

            loss = loss / gacc
            scaler.scale(loss).backward()

            if step % gacc == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            running += float(loss.item()) * xb.size(0) * gacc

        running /= len(train_ds)
        sched.step()
        print(f"[Epoch {ep}] loss={running:.6f} lr={sched.get_last_lr()[0]:.2e}")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "feature_cols": feature_cols}, cfg["model_path"])
    with open(cfg["scaler_path"], "wb") as f: pickle.dump(scalers, f)
    print("OK")

if __name__ == "__main__":
    main()
