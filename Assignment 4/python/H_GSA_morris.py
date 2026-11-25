# -*- coding: utf-8 -*-
"""
Task 2.2 - Global sensitivity analysis with the Morris method

morrisGSA estimates the elementary effects of the model by using the
method proposed by Morris (1991) and subsequently modified by Campolongo
et al. (2007)  (course code, adapted to Model 4)

INPUT  - n_trj     = number of trajectories to be analyzed
       - par_info  = structured array with name, min and max of the parameters
                     (param_ranges in A_main.py: 9 model parameters + flowThr)
       - inputData = measured data (flow and NH4)
       - wet       = fixed wet-weather evaluation mask (True/False per time step)
OUTPUT - mu_star   = mean of the absolute values of the sensitivity indices
       - sigma     = standard deviation of the sensitivity indices
       - mu        = mean of the sensitivity indices (with sign)
       - par_sample, obj_fun = all parameter sets and their J values
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydoe import lhs

from B_Model import NH4inletModel4
from C_objective_function import calculate_nse, wet_mask


def morrisGSA(n_trj, par_info, inputData, wet):

    ## generate trajectories across the parameter space
    n_par = len(par_info)            # number of parameters (10)
    n_sim = n_trj * (n_par + 1)      # number of model runs
    dx = 1 / n_trj

    # Latin Hypercube Sampling of the trajectory starting points
    # (option "cm" = centermaximin, as in the course code)
    trj_start_norm = lhs(n_par, samples=n_trj, criterion='cm')
    trj_start = par_info['min'] + trj_start_norm * (np.array([par_info['max'] - par_info['min']]))
    par_sample = np.empty([0, n_par])   # all points of all trajectories

    for i in range(n_trj):
        trj_point = trj_start[i, :] * (1 + np.tril(np.ones([n_par + 1, n_par]), -1) * dx)
        par_sample = np.append(par_sample, trj_point, axis=0)

    ## run the GSA analysis
    obs = inputData.loc[wet, "nh4_imputed"]
    obj_fun = np.zeros(n_sim)           # J = 1 - NSE_wet for each parameter set

    for i in range(n_sim):
        print(f"analyzing parameter set {i + 1} out of {n_sim}")
        p = par_sample[i, :]
        out = NH4inletModel4(p[:9], inputData, p[9])      # 9 model parameters + flowThr
        nse = calculate_nse(obs, out.loc[wet, "simNH4conc"])
        obj_fun[i] = 1 - nse if np.isfinite(nse) else np.nan

    ## calculate indices (relative elementary effects, as in the course code)
    d = np.zeros([n_trj, n_par])
    for i in range(n_trj):
        for j in range(n_par):
            d[i, j] = (obj_fun[i*(n_par+1) + (j+1)] - obj_fun[i*(n_par+1) + j]) / obj_fun[i*(n_par+1) + j] / dx

    mu = np.nanmean(d, axis=0)
    sigma = np.nanstd(d, axis=0)
    mu_star = np.nanmean(np.abs(d), axis=0)

    return mu_star, sigma, mu, par_sample, obj_fun


def plot_morris(mu, mu_star, sigma, names, out_dir=None):

    n_par = len(names)
    x_par = np.arange(n_par)

    def save(fig, file_name):
        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, file_name), dpi=300, bbox_inches="tight")

    # 1) bars of mu (with sign)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x_par, mu)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_par)
    ax.set_xticklabels(names)
    ax.set_xlabel("Parameter")
    ax.set_ylabel(r"Mean elementary effect $\mu$ [-]")
    ax.set_title(r"Morris GSA – mean elementary effect $\mu$")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "Morris_mu_bar.pdf")

    # 2) bars of mu star (importance)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x_par, mu_star)
    ax.set_xticks(x_par)
    ax.set_xticklabels(names)
    ax.set_xlabel("Parameter")
    ax.set_ylabel(r"Mean absolute elementary effect $\mu^*$ [-]")
    ax.set_title(r"Morris GSA – parameter importance $\mu^*$")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "Morris_mu_star_bar.pdf")

    # 3) mu vs sigma
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(mu, sigma, "o")
    for i in range(n_par):
        ax.annotate(names[i], (mu[i], sigma[i]), textcoords="offset points", xytext=(4, 4))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Mean elementary effect $\mu$ [-]")
    ax.set_ylabel(r"Standard deviation of elementary effects $\sigma$ [-]")
    ax.set_title(r"Morris GSA – $\mu$ vs. $\sigma$")
    ax.grid(alpha=0.3)
    save(fig, "Morris_mu_vs_sigma.pdf")

    # 4) mu star vs sigma (red line: sigma = mu*)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vals = np.concatenate([mu_star, sigma])
    vals = vals[vals > 0]                          # log scale cannot show 0
    lo, hi = 0.5 * vals.min(), 2 * vals.max()
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, label=r"$\sigma = \mu^*$")
    ax.plot(mu_star, sigma, "o")
    for i in range(n_par):
        ax.annotate(names[i], (mu_star[i], sigma[i]), textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Mean absolute elementary effect $\mu^*$ [-]")
    ax.set_ylabel(r"Standard deviation of elementary effects $\sigma$ [-]")
    ax.set_title(r"Morris GSA – importance $\mu^*$ vs. non-linearity/interaction $\sigma$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "Morris_mu_star_vs_sigma_log.pdf")

    plt.show()

# ---------------------------------------------------------------
# RUN THE GSA (only when this file is started directly)
# ---------------------------------------------------------------

if __name__ == "__main__":

    from A_main import inputData, param_ranges, evalThr, lag_h

    n_trj = 20                  # number of trajectories -> n_trj * 11 model runs
    np.random.seed(1)           # same random trajectories every time (reproducible)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    # fixed wet-weather evaluation period (same as Task 1.2 and the OAT)
    wet = wet_mask(inputData, evalThr, lag_h)

    mu_star, sigma, mu, par_sample, obj_fun = morrisGSA(n_trj, param_ranges, inputData, wet)

    # results table
    names = list(param_ranges['name'])
    df = pd.DataFrame({"parameter": names, "mu": mu, "mu_star": mu_star, "sigma": sigma})
    df["sigma/mu_star"] = df["sigma"] / df["mu_star"]
    df["rank"] = df["mu_star"].rank(ascending=False, method="min").astype(int)

    print(f"\nMorris GSA: {n_trj} trajectories, {len(obj_fun)} model runs")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv(os.path.join(out_dir, "Morris_results.csv"), sep=";", index=False)

    # all parameter sets and J values (e.g. for later checks)
    samples = pd.DataFrame(par_sample, columns=names)
    samples["J"] = obj_fun
    samples.to_csv(os.path.join(out_dir, "Morris_samples.csv"), sep=";", index=False)

    plot_morris(mu, mu_star, sigma, names, out_dir)