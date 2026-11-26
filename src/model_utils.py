"""helpers for task 3 and the side scripts"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from NH4models import NH4inletModel4
import objective_function as of
from data_utils import FLOW_THR, ROOT

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# bigger fonts
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                     "lines.linewidth": 1.6})


# optimiser works on params scaled to [0, 1]
def to_phys(u, bounds):
    lo, hi = np.array(bounds).T
    return lo + np.asarray(u) * (hi - lo)

def to_unit(x, bounds):
    lo, hi = np.array(bounds).T
    return (np.asarray(x) - lo) / (hi - lo)


def simulate4(p_full, data):
    # 9 params in physical units (g/hr, m3), or 10 with Q_thr [m3/hr] as the last one
    p_full = np.asarray(p_full, float)
    thr = p_full[9] if len(p_full) > 9 else FLOW_THR
    with np.errstate(all="ignore"):      # unstable params overflow, handled in the objective
        return NH4inletModel4(p_full[:9], data, thr)["simNH4conc"]


def evaluate(data, p_phys):
    # NSE, RMSE, MARE on scored points: all and wet only
    sim = np.asarray(simulate4(p_phys, data))
    rows = []
    for label, m in [("all", data["scored"].values), ("wet", (data["scored"] & data["wet"]).values)]:
        rows.append((label, of.NSE(data["nh4"][m], sim[m]), of.RMSE(data["nh4"][m], sim[m]),
                     of.MARE(data["nh4"][m], sim[m])))
    return pd.DataFrame(rows, columns=["points", "NSE", "RMSE g/m3", "MARE"]).round(3)


def plot_fit(data, p_phys, title, fname):
    # observed vs simulated, warm-up hidden
    sim = simulate4(p_phys, data)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5),
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(data["time"], data["nh4"].where(data["valid"]), "k", lw=0.8, label="observed")
    ax[0].plot(data["time"], data["nh4"].where(data["discard"]), color="tab:orange", lw=1.2,
               label="gap-filled (not scored)")
    ax[0].plot(data["time"], np.asarray(sim, float)[:] * np.where(data["warmup"], np.nan, 1.0),
               "tab:red", lw=0.9, label="simulated (Model 4)")
    ax[0].axvspan(data["time"].iloc[0], data["time"][data["warmup"]].iloc[-1], color="grey", alpha=0.2, lw=0)  # warm-up
    ax[0].set_ylabel("NH4 [g/m3]"); ax[0].set_title(title, pad=34)
    ax[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)   # above the data
    ax[1].plot(data["time"], data["flow"], "tab:blue", lw=0.7)
    ax[1].axhline(FLOW_THR, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("Flow [m3/hr]")
    for a in ax:
        a.grid(alpha=0.3)
    loc = mdates.AutoDateLocator(maxticks=6)
    ax[1].xaxis.set_major_locator(loc); ax[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, fname), dpi=200); plt.close(fig)


def plot_comparison(data, sims, title, fname):
    # observed vs several simulations (dict name -> array), wet periods shaded
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6.5), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(data["time"], data["nh4"].where(data["valid"]), "k", lw=1.2, label="observed")
    for name, s in sims.items():
        ax[0].plot(data["time"], np.where(data["warmup"], np.nan, np.asarray(s, float)), lw=1.3, label=name)
    for a in ax:
        a.fill_between(data["time"], 0, 1, where=data["wet"].values, transform=a.get_xaxis_transform(),
                       color="tab:blue", alpha=0.10, lw=0)                         # wet periods
        a.axvspan(data["time"].iloc[0], data["time"][data["warmup"]].iloc[-1], color="grey", alpha=0.25, lw=0)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("NH4 [g/m3]")
    ax[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)   # above the data
    ax[0].set_title(title, pad=34)
    ax[1].plot(data["time"], data["flow"], "tab:blue", lw=1.0)
    ax[1].axhline(FLOW_THR, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("Flow [m3/hr]")
    loc = mdates.AutoDateLocator(maxticks=6)
    ax[1].xaxis.set_major_locator(loc); ax[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, fname), dpi=200); plt.close(fig)


def position_in_range(p_opt, bounds, names):
    # 0 % / 100 % = on the border of the range
    rows = [(n, v, lo, hi, 100 * (v - lo) / (hi - lo)) for n, v, (lo, hi) in zip(names, p_opt, bounds)]
    return pd.DataFrame(rows, columns=["param", "value", "lo", "hi", "% of range"]).round(3)
