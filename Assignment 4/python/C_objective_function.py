import numpy as np
from B_Model import NH4inletModel4


# ============================================================================
# WET-WEATHER EVALUATION PERIOD
# Q > evalThr, plus lag_h hours after the flow falls below evalThr again
# ============================================================================

def wet_mask(inputData, evalThr=5000, lag_h=6):
    above = inputData["flow_smoothed"] > evalThr
    dt = inputData["time"].diff().dt.total_seconds().median() / 3600  # [h]
    n_lag = int(round(lag_h / dt))                                    # lag in time steps
    # True if the flow was above evalThr at any time in the last lag_h hours
    return above.rolling(n_lag + 1, min_periods=1).max().astype(bool)


# ============================================================================
# OBJ A - NASH-SUTCLIFFE EFFICIENCY
# ============================================================================

def calculate_nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[valid]
    simulated = simulated[valid]
    if len(observed) < 2:
        return np.nan
    numerator = np.sum((observed - simulated)**2)
    denominator = np.sum((observed - np.mean(observed))**2)
    if denominator == 0:
        return np.nan
    return 1 - numerator / denominator


# ============================================================================
# OBJ B - RELATIVE EVENT LOAD ERROR
# ============================================================================

def calculate_event_load_error(observed, simulated, dt):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)

    # Flux [g/h] -> total mass [kg]
    mass_obs = np.sum(observed * dt) / 1000
    mass_sim = np.sum(simulated * dt) / 1000
    if mass_obs == 0:
        return np.nan, np.nan, np.nan
    error = (mass_sim - mass_obs) / mass_obs * 100
    return mass_obs, mass_sim, error


# ============================================================================
# OPTIMIZATION WRAPPER - OBJ A
# ============================================================================

def objective_model4(param, inputData, flowThr, evalThr=5000, lag_h=6):
    result = NH4inletModel4(param, inputData, flowThr)

    # Fixed wet-weather evaluation period (independent of flowThr)
    wet = wet_mask(inputData, evalThr, lag_h)
    nse = calculate_nse(
        inputData.loc[wet, "nh4_imputed"],
        result.loc[wet, "simNH4conc"]
    )
    return 1 - nse if np.isfinite(nse) else 1e12