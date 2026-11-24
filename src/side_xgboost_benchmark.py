"""
side script: xgboost benchmark for NH4 concentration (not part of the assignment)
same split and metrics as task3, features use only time and flow (no measured NH4)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from data_utils import load_and_split, FLOW_THR, RESULTS_DIR
from objective_function import NSE, RMSE
from model_utils import simulate4, plot_comparison

cal, val, _ = load_and_split()

TARGET = "nh4"
PLOT_METHOD = "nelder-mead"   # method drawn in the figures (table has all)
STEP_H = cal["time"].diff().dt.total_seconds().median() / 3600     # time step [h]


def make_features(d):
    t = d["time"].dt.hour / 24 + d["time"].dt.minute / 1440
    f = pd.DataFrame({"sin1": np.sin(2 * np.pi * t), "cos1": np.cos(2 * np.pi * t),
                      "sin2": np.sin(4 * np.pi * t), "cos2": np.cos(4 * np.pi * t),
                      "flow": d["flow"]})
    for lag_h in (0.5, 1, 2, 6):                                  # past flow only
        f[f"flow_lag{lag_h}h"] = d["flow"].shift(round(lag_h / STEP_H)).bfill()
    f["flow_mean6h"] = d["flow"].rolling(round(6 / STEP_H), min_periods=1).mean()
    # hours since last exceedance of the threshold
    above = (d["flow"] > FLOW_THR).values
    tsle = np.zeros(len(d)); tsle[0] = 24 * 30
    for i in range(1, len(d)):
        tsle[i] = 0 if above[i] else tsle[i - 1] + STEP_H
    f["h_since_event"] = np.minimum(tsle, 24 * 30)
    return f


X_cal, X_val = make_features(cal), make_features(val)
m_cal, m_val = cal["scored"].values, val["scored"].values   # same scored points as Model 4

# last 20 % of cal = early stopping set
n_fit = int(0.8 * len(cal))
fit = m_cal & (np.arange(len(cal)) < n_fit)
stop = m_cal & (np.arange(len(cal)) >= n_fit)

model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=30, random_state=0)
model.fit(X_cal[fit], cal.loc[fit, TARGET], eval_set=[(X_cal[stop], cal.loc[stop, TARGET])],
          verbose=False)
print(f"best iteration: {model.best_iteration}")

# compare with model 4 for each p_best_<method>.npy
sims = {"XGBoost": {"cal": model.predict(X_cal), "val": model.predict(X_val)}}
for f in sorted(RESULTS_DIR.glob("p_best_*.npy")):
    p = np.load(f)
    sims[f"Model 4 ({f.stem[7:]})"] = {"cal": np.asarray(simulate4(p, cal)), "val": np.asarray(simulate4(p, val))}

rows = []
for name, s in sims.items():
    for setname, d, m in [("cal", cal, m_cal), ("val", val, m_val)]:
        for sub, mm in [("all", m), ("wet", m & d["wet"].values)]:
            rows.append((name, setname, sub, NSE(d.loc[mm, TARGET], s[setname][mm]),
                         RMSE(d.loc[mm, TARGET], s[setname][mm])))
res = pd.DataFrame(rows, columns=["model", "set", "points", "NSE", "RMSE g/m3"]).round(3)
print(res.to_string(index=False))
res.to_csv(RESULTS_DIR / "side_xgboost_comparison.csv", index=False)

# feature importance
imp = pd.Series(model.feature_importances_, index=X_cal.columns).sort_values()
print("\nfeature importance:\n", imp.round(3).to_string())

# comparison figures, xgboost vs model 4 (PLOT_METHOD)
for setname, d in [("cal", cal), ("val", val)]:
    plot_sims = {"XGBoost": sims["XGBoost"][setname], f"Model 4 ({PLOT_METHOD})": sims[f"Model 4 ({PLOT_METHOD})"][setname]}
    plot_comparison(d, plot_sims, {"cal": "Calibration period", "val": "Validation period"}[setname] + ": XGBoost vs Model 4",
                    f"side_comparison_{setname}.png")
