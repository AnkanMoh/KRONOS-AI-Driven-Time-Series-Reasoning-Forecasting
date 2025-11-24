import json, numpy as np, pandas as pd, yfinance as yf, yaml, os

TARGETS = ["Open","Close","Low","High","Volume"]

def _clean(df):
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    cols = [str(c).strip().title() for c in df.columns]; df = df.copy(); df.columns = cols
    out = pd.DataFrame(index=df.index)
    for name in ["Open","High","Low","Close","Volume"]:
        ser = df[name] if name in df.columns else pd.Series(index=df.index, dtype=float)
        out[name] = ser
    return out.dropna(subset=["Open","High","Low","Close","Volume"])

def _pos(index, d):
    loc = index.get_loc(d)
    if isinstance(loc, slice): return int(range(loc.start, loc.stop)[-1])
    if isinstance(loc, (np.ndarray, list)): return int(loc[-1])
    return int(loc)

def _scalar(a, r, c): return float(a[int(r), int(c)])

def _cls(y_t, y_tp1):
    if y_tp1 > y_t: return 1
    if y_tp1 < y_t: return -1
    return 0

def _macro(trues, preds):
    classes = [-1,0,1]
    trues = np.asarray(trues); preds = np.asarray(preds)
    acc = float((trues==preds).mean()) if len(trues) else 0.0
    P=R=F=0.0
    for c in classes:
        tp = np.sum((preds==c)&(trues==c)); fp = np.sum((preds==c)&(trues!=c)); fn = np.sum((preds!=c)&(trues==c))
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0; r = tp/(tp+fn) if (tp+fn)>0 else 0.0; f = (2*p*r)/(p+r) if (p+r)>0 else 0.0
        P+=p; R+=r; F+=f
    return float(P/3), float(R/3), float(F/3), acc

def main():
    with open("config.yaml","r") as f: cfg = yaml.safe_load(f)
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    with open(cfg["predictions_path"], "r") as f: preds = json.load(f)
    v_end = pd.to_datetime(cfg["valid_end_date"])
    rows = []
    for t in sorted(preds.keys()):
        gt = yf.download(t, start=cfg["valid_start_date"], end=(v_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False, group_by="column")
        if gt is None or gt.empty: continue
        gt = _clean(gt); gt.index = pd.to_datetime(gt.index).tz_localize(None); gt = gt.sort_index(); gt = gt[~gt.index.duplicated(keep="last")]
        p_dates = [pd.to_datetime(d) for d in preds[t].keys()]
        idx = sorted(set(gt.index) & set(p_dates))
        if not idx: continue
        a = gt[["Open","Close","Low","High","Volume"]].to_numpy()
        m = {"Open":0,"Close":1,"Low":2,"High":3,"Volume":4}
        for target in TARGETS:
            j = m[target]
            y_true, y_pred, dir_true, dir_pred = [], [], [], []
            for d in idx:
                try: pos = _pos(gt.index, d)
                except KeyError: continue
                if pos<=0: continue
                gt_today = _scalar(a, pos, j); prev_val = _scalar(a, pos-1, j)
                pj = preds[t][str(d.date())].get(target)
                if pj is None: continue
                pj = float(pj)
                y_true.append(gt_today); y_pred.append(pj)
                dir_true.append(_cls(prev_val, gt_today)); dir_pred.append(_cls(prev_val, pj))
            if not y_true: continue
            y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
            mse = float(np.mean((y_true - y_pred)**2))
            P,R,F,Acc = _macro(dir_true, dir_pred) if dir_true and dir_pred else (0,0,0,0)
            rows.append([t, target, round(mse,6), round(P,6), round(R,6), round(F,6), round(Acc,6)])
    out = cfg["evaluation_csv"]
    if rows:
        df = pd.DataFrame(rows, columns=["Ticker","Target","MSE","Precision","Recall","F1","Accuracy"])
        order = {k:i for i,k in enumerate(["Open","Close","Low","High","Volume"])}
        df = df.sort_values(by=["Ticker","Target"], key=lambda s: s.map(order) if s.name=="Target" else s)
        df.to_csv(out, index=False); print("OK")
    else:
        print("OK")

if __name__ == "__main__":
    main()
