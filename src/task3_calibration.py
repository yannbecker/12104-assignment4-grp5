"""
Assignment 4
Task 3: calibration of model 4

params: [a0, a1, b1, a2, b2, c1, c2, c3, V, Qthr]

steps:
A Fourier coefficients -> dry calibration data only (model 0)
B c1, c2, c3, V, Qthr  -> full cal set, Fourier fixed
C robustness (different starting points)
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo

from NH4models import NH4inletModel0
import objective_function as of
from data_utils import load_and_split, RESULTS_DIR
from model_utils import to_phys, to_unit, simulate4, evaluate, plot_fit, position_in_range

# settings
# optimiser, choose one
#   local: "Nelder-Mead", "L-BFGS-B", "Powell", "TNC", "SLSQP"
#   global: "differential_evolution", "dual_annealing", "shgo"
# outputs are tagged with the method name
METHOD = sys.argv[1] if len(sys.argv) > 1 else "L-BFGS-B"     # python task3_calibration.py L-BFGS-B
TAG = METHOD.lower()
GLOBAL_METHODS = ("differential_evolution", "dual_annealing", "shgo")
TOL = 1e-4
NM_MAXITER = 5000          # simplex stops earlier if converged

NAMES = ["a0", "a1", "b1", "a2", "b2", "c1", "c2", "c3", "V", "Qthr"]

# Bounds
# new "Your min / Your max" ranges (Fourier in kg/hr)
BOUNDS_FOURIER = [(95, 135), (-40, -5), (-50, -1.5), (-30, 0), (-5, 25)]
BOUNDS_WET = [(0.0, 1.0),      # c1 max dilution
              (0.0, 1.0),      # c2 timing of max dilution
              (0.1, 20.0),     # c3 width of dilution
              (5000., 20000.), # V buffer volume [m3]
              (4000., 6000.)]  # Q_thr wet-weather flow threshold [m3/hr]
X0_FOURIER = np.array([124.6, -32.3, -15.6, -5.88, 17.4], float) # initial condition, fourier
X0_WET = np.array([0.45, 0.5, 8.0, 11500., 5000.]) # initial condition, wet params

cal, val, cal_dry = load_and_split()


# objective functions (minimised) = 1 - NSE
def obj_fourier(u, data):
    # step A: model 0 on dry data, u = scaled params
    p = to_phys(u, BOUNDS_FOURIER) * 1e3                     # kg/hr to g/hr
    sim = NH4inletModel0(p, data)["simNH4conc"]
    return 1 - of.NSE(data["nh4"], sim)


def obj_wet(u, data, fourier_phys):
    # step B: free = c1, c2, c3, V, Qthr (scaled), Fourier fixed
    sim = simulate4(np.concatenate([fourier_phys, to_phys(u, BOUNDS_WET)]), data)
    m = data["scored"].values
    s = np.asarray(sim)[m]
    if not np.all(np.isfinite(s)):
        return 1e10 #penalise diverging values
    return 1 - of.NSE(data["nh4"].values[m], s)


def calibrate(obj, x0, bounds, args, maxiter=1000):
    unit = [(0, 1)] * len(bounds)
    if METHOD == "differential_evolution":     # global: ignore x0, small settings (slow model)
        res = differential_evolution(obj, unit, args=args, seed=1, maxiter=min(maxiter, 30),
                                     popsize=8, tol=TOL, polish=False)
    elif METHOD == "dual_annealing":
        res = dual_annealing(obj, unit, args=args, seed=1, maxiter=min(maxiter, 200))
    elif METHOD == "shgo":
        res = shgo(obj, unit, args=args, n=64, iters=2, sampling_method="sobol")
    else:
        if METHOD == "Nelder-Mead":
            maxiter = NM_MAXITER                  # longer limit for simplex
        res = minimize(obj, to_unit(x0, bounds), args=args, method=METHOD,
                       bounds=unit, tol=TOL, options={"maxiter": maxiter})
    print(f"{METHOD}: 1-NSE={res.fun:.4f}  nfev={res.nfev}  {res.message}")
    return res.fun, to_phys(res.x, bounds)


#  MAIN
if __name__ == "__main__":
    #Step A
    _, fourier_kg = calibrate(obj_fourier, X0_FOURIER, BOUNDS_FOURIER, (cal_dry,))
    fourier_phys = fourier_kg * 1e3

    # Step B
    _, wet = calibrate(obj_wet, X0_WET, BOUNDS_WET, (cal, fourier_phys))
    p_best = np.concatenate([fourier_phys, wet])               # physical units, 10 values

    print("\nBest parameters:")
    print(pd.Series(p_best, index=NAMES).round(3).to_string())
    pos = pd.concat([position_in_range(fourier_kg, BOUNDS_FOURIER, NAMES[:5]),     # Fourier in kg/hr
                     position_in_range(wet, BOUNDS_WET, NAMES[5:])])
    print(pos.to_string(index=False))
    pos.insert(0, "method", METHOD)
    pos.to_csv(RESULTS_DIR / f"task3_params_{TAG}.csv", index=False)       # best values + position in range

    print("\ncalibration performance:")
    perf = evaluate(cal, p_best)
    print(perf.to_string(index=False))
    perf.insert(0, "set", "calibration"); perf.insert(0, "method", METHOD)
    perf.to_csv(RESULTS_DIR / f"task3_metrics_{TAG}.csv", index=False)

    np.save(RESULTS_DIR / f"p_best_{TAG}.npy", p_best)
    plot_fit(cal, p_best, f"Task 3.2 - calibration ({METHOD})", f"task3_2_calibration_{TAG}.png")

    # task 3.5: robustness to starting point (step B), skipped for global methods
    if METHOD in GLOBAL_METHODS:
        sys.exit(0)
    rng = np.random.default_rng(0)
    lo, hi = np.array(BOUNDS_WET).T
    runs = []
    for k in range(8):
        f, x = calibrate(obj_wet, rng.uniform(lo, hi), BOUNDS_WET, (cal, fourier_phys), maxiter=400)
        runs.append(np.r_[f, x])
    runs = pd.DataFrame(runs, columns=["1-NSE"] + NAMES[5:])
    runs.to_csv(RESULTS_DIR / f"task3_5_multistart_{TAG}.csv", index=False)
    print(runs.describe().round(3).to_string())
    # todo: repeat with narrower / wider BOUNDS_WET
