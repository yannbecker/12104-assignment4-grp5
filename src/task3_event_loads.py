"""OBJ B: NH4 load per wet event (flow > 5000 m3/h), observed vs simulated (model 4, L-BFGS-B params)"""
import numpy as np, pandas as pd
from data_utils import load_and_split, RESULTS_DIR, FLOW_THR
from model_utils import simulate4

p = np.load(RESULTS_DIR / "p_best_l-bfgs-b.npy")
cal, val, _ = load_and_split(verbose=False)
rows = []
for name, d in [("cal", cal), ("val", val)]:
    sim = np.asarray(simulate4(p, d), float)
    dt = d["time"].diff().dt.total_seconds().median() / 3600          # h per step
    above = (d["flow"] > FLOW_THR).values
    ev = np.cumsum(above & ~np.r_[False, above[:-1]]) * above       # event id
    for k in sorted(set(ev) - {0}):
        m = ev == k
        q = d["flow"].values[m]
        lo = (q * d["nh4"].values[m]).sum() * dt / 1e3                # kg
        ls = (q * sim[m]).sum() * dt / 1e3
        rows.append((name, d["time"][m].iloc[0], m.sum() * dt, lo, ls, 100 * (ls - lo) / lo,
                     d["discard"].values[m].mean() * 100, d["nh4"].values[m].mean(), sim[m].mean()))
r = pd.DataFrame(rows, columns=["set", "start", "duration_h", "obs_load_kg", "sim_load_kg",
                                "error_%", "discarded_%", "obs_mean_nh4", "sim_mean_nh4"]).round(1)
print(r.to_string(index=False))
for s in ("cal", "val"):
    x = r[r.set == s]
    print(s, "total obs", x.obs_load_kg.sum().round(0), "sim", x.sim_load_kg.sum().round(0),
          "mean |err| %", x["error_%"].abs().mean().round(1))
r.to_csv(RESULTS_DIR / "task3_event_loads_l-bfgs-b.csv", index=False)
