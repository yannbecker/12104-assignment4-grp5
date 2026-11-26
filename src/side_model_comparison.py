"""side script: compare model 0/1/3/4 (and joint-fit variants), same split + bounds + metrics as task3"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from NH4models import NH4inletModel0, NH4inletModel1, NH4inletModel3, NH4inletModel4
import objective_function as of
from data_utils import load_and_split, RESULTS_DIR, FLOW_THR
from task3_calibration import BOUNDS_FOURIER, X0_FOURIER, obj_fourier, calibrate  # runs METHOD=L-BFGS-B default
from model_utils import to_phys, to_unit

cal, val, cal_dry = load_and_split(verbose=False)
BW = dict(c1=(0, 1), c2=(0, 1), c3=(0.1, 20), V=(5000., 20000.))
X0 = dict(c1=0.45, c2=0.5, c3=8.0, V=11500.)

def sim(model, p, d):
    p = np.asarray(p, float)
    with np.errstate(all="ignore"):
        if model == "M0": return np.asarray(NH4inletModel0(p[:5], d)["simNH4conc"])
        if model == "M1": return np.asarray(NH4inletModel1(p[:6], d)["simNH4conc"])
        if model == "M3": return np.asarray(NH4inletModel3(p[:8], d, FLOW_THR)["simNH4conc"])
        if model == "M4": return np.asarray(NH4inletModel4(p[:9], d, FLOW_THR)["simNH4conc"])

# wet params per model, in the order the model expects after the 5 Fourier values
WET = {"M0": [], "M1": ["V"], "M3": ["c1", "c2", "c3"], "M4": ["c1", "c2", "c3", "V"]}

def score(d, s, wet=False):
    m = (d["scored"] & d["wet"]).values if wet else d["scored"].values
    return of.NSE(d["nh4"].values[m], s[m]), of.RMSE(d["nh4"].values[m], s[m])

def obj(u, model, d, bounds, fixed_f, wet_only=False):
    p = to_phys(u, bounds)
    p = np.r_[fixed_f, p] if fixed_f is not None else p
    p[:5] = p[:5] if fixed_f is not None else p[:5]
    if fixed_f is None: p = np.r_[p[:5] * 1e3, p[5:]]
    s = sim(model, p, d)
    m = (d["scored"] & d["wet"]).values if wet_only else d["scored"].values
    if not np.all(np.isfinite(s[m])): return 1e10
    return 1 - of.NSE(d["nh4"].values[m], s[m])

fourier_kg = minimize(obj_fourier, to_unit(X0_FOURIER, BOUNDS_FOURIER), args=(cal_dry,), method="L-BFGS-B",
                      bounds=[(0, 1)] * 5, tol=1e-4).x
fourier_kg = to_phys(fourier_kg, BOUNDS_FOURIER); fF = fourier_kg * 1e3
rng = np.random.default_rng(0)
rows = []

def run(label, model, joint, wet_only=False):
    names = WET[model]
    bw = [BW[n] for n in names]
    bounds = (BOUNDS_FOURIER if joint else []) + bw
    if not bounds:
        p = fF.copy()
    else:
        starts = [np.r_[to_unit(X0_FOURIER, BOUNDS_FOURIER) if joint else [], to_unit([X0[n] for n in names], bw)]]
        starts += [rng.uniform(0, 1, len(bounds)) for _ in range(5)]
        best = min((minimize(obj, s, args=(model, cal, bounds, None if joint else fF, wet_only),
                             method="L-BFGS-B", bounds=[(0, 1)] * len(bounds), tol=1e-5) for s in starts),
                   key=lambda r: r.fun)
        x = to_phys(best.x, bounds)
        p = np.r_[x[:5] * 1e3, x[5:]] if joint else np.r_[fF, x]
    for sname, d in (("cal", cal), ("val", val)):
        s = sim(model, p, d)
        for pts, w in (("all", False), ("wet", True)):
            n, r = score(d, s, w)
            rows.append((label, sname, pts, round(n, 3), round(r, 2)))
    print(label, np.round(p, 2), flush=True)

for mdl in ("M0", "M1", "M3", "M4"):
    run(f"{mdl} stepA+B", mdl, joint=False)
for mdl in ("M1", "M3", "M4"):
    run(f"{mdl} joint", mdl, joint=True)
run("M4 joint, wet-only obj", "M4", joint=True, wet_only=True)

r = pd.DataFrame(rows, columns=["model", "set", "points", "NSE", "RMSE"])
t = r.pivot_table(index="model", columns=["set", "points"], values="NSE", sort=False)
print(t.to_string())
r.to_csv(RESULTS_DIR / "side_model_comparison.csv", index=False)
