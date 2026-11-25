"""
Task 2.1 - Local sensitivity analysis (OAT) for Model 4

Point in parameter space : baseline from A_main.py (param + flowThr)
Objective function       : J = 1 - NSE_wet   (wet = Q > evalThr + lag_h hours after)
Perturbation             : +10 % and -10 % of the baseline value, one parameter at a time
Sensitivity index        : S = ((J - J0) / J0) / (+/- 0.10)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from A_main import inputData, param, flowThr, evalThr, lag_h
from B_Model import NH4inletModel4
from C_objective_function import calculate_nse, wet_mask


# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------

change = 0.10
out_dir = "output"

names = ["a0", "a1", "b1", "a2", "b2", "c1", "c2", "c3", "V", "flowThr"]

# baseline: the 9 model parameters + flowThr as 10th value
base = np.append(param, flowThr).astype(float)
k = len(base)

# fixed evaluation period (same for all runs)
wet = wet_mask(inputData, evalThr, lag_h)
obs = inputData.loc[wet, "nh4_imputed"]


def J(theta):
    """Run Model 4 and return J = 1 - NSE_wet. theta = 9 params + flowThr."""
    result = NH4inletModel4(theta[:9], inputData, theta[9])
    nse = calculate_nse(obs, result.loc[wet, "simNH4conc"])
    return 1 - nse


# ---------------------------------------------------------------
# PARAMETER SETS (10 x 10): each row changes one parameter
# ---------------------------------------------------------------

P_plus = base * (1 + change * np.eye(k))
P_minus = base * (1 - change * np.eye(k))


# ---------------------------------------------------------------
# MODEL RUNS (1 baseline + 2 x 10)
# ---------------------------------------------------------------

J0 = J(base)
J_plus = np.array([J(row) for row in P_plus])
J_minus = np.array([J(row) for row in P_minus])

S_plus = ((J_plus - J0) / J0) / (+change)
S_minus = ((J_minus - J0) / J0) / (-change)


# ---------------------------------------------------------------
# RESULTS TABLE -> CSV
# ---------------------------------------------------------------

df = pd.DataFrame({
    "parameter": names,
    "baseline": base,
    "value_plus10": np.diag(P_plus),
    "value_minus10": np.diag(P_minus),
    "J0": J0,
    "J_plus10": J_plus,
    "J_minus10": J_minus,
    "S_plus10": S_plus,
    "S_minus10": S_minus,
})
df["S_abs_mean"] = (df["S_plus10"].abs() + df["S_minus10"].abs()) / 2
# keep the parameter order from A_main.py (param array, then flowThr);
# the rank column still shows which parameter is most sensitive
df["rank"] = df["S_abs_mean"].rank(ascending=False, method="min").astype(int)

os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "OAT_results.csv"), sep=";", index=False)

print(f"\nBaseline: NSE_wet = {1 - J0:.4f}, J0 = {J0:.4f}  "
      f"(wet = Q > {evalThr} m3/h + {lag_h} h, n = {wet.sum()})")
print(df[["rank", "parameter", "J_plus10", "J_minus10",
          "S_plus10", "S_minus10", "S_abs_mean"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# plot paramter array with baseline and ±10 % values
# pd.set_option("display.max_columns", None)   # show all columns
# pd.set_option("display.width", 200)          # don't wrap lines
# print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------
# PLOT: |S| for +10 % and -10 %, in the parameter order of A_main.py
# ---------------------------------------------------------------

x = np.arange(k)
w = 0.38

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - w / 2 - 0.01, df["S_plus10"].abs(), w, color="#2a78d6", label="+10 %")
ax.bar(x + w / 2 + 0.01, df["S_minus10"].abs(), w, color="#eb6834", label="−10 %")

ax.set_xticks(x)
ax.set_xticklabels(df["parameter"])
ax.set_xlabel("Parameter")
ax.set_ylabel("|S| [-]")
ax.set_title("Local sensitivity (OAT, ±10 %)")
ax.legend(frameon=False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "OAT_sensitivity.pdf"), dpi=200)
plt.show()