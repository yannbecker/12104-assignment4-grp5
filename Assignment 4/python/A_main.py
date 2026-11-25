import numpy as np
import pandas as pd

from B_Model import NH4inletModel4
from D_plots import plot_model
from D_plots import plot_wet_day
from C_objective_function import calculate_nse, calculate_event_load_error, wet_mask


# ---------------------------------------
# SHARED SETTINGS (also imported by F_ and G_ scripts)

inputData = pd.read_csv("data/data_preprocessed.csv", sep=";", parse_dates=["time"], dayfirst=True)

# plotting period (None = whole dataset)
start_date = None#"2018-10-07 00:00:00"
end_date   = None#"2018-10-08 00:00:00"

param = np.array([
    # Dry-weather NH4 load (Fourier series) [g/h]
    124600,   # a0 - param[0]   Mean NH4 load
    -32300,   # a1 - param[1]   First harmonic (sine)
    -15600,   # b1 - param[2]   First harmonic (cosine)
    -5880,    # a2 - param[3]   Second harmonic (sine)
    17400,    # b2 - param[4]   Second harmonic (cosine)
    # Rainfall-related dilution [-]
    0.45,     # c1 - param[5]   Dilution magnitude
    0.5,      # c2 - param[6]   Dilution timing parameter
    8.0,      # c3 - param[7]   Dilution width parameter
    # Upstream buffering
    11500     # V  - param[8]   Total volume [m3]
])

flowThr = 5000  # Q  - param[9] flow [m3/h]  model parameter: wet-weather flow threshold
evalThr = 5000  # evaluation threshold for the objective function [m3/h] (kept fixed)
lag_h   = 6     # hours after Q < evalThr that still count as wet weather [h]

param_ranges = np.array([
    ('a0',      95000, 135000),
    ('a1',     -40000,  -5000),
    ('b1',     -50000,  -1500),
    ('a2',     -30000,      0),
    ('b2',      -5000,  25000),
    ('c1',          0,      1),
    ('c2',          0,      1),
    ('c3',        0.1,     20),
    ('V',        5000,  20000),
    ('flowThr',  4000,   6000)],
    dtype=[('name', 'U10'), ('min', 'f4'), ('max', 'f4')])

# ---------------------------------------
# ONLY RUNS WITH "python A_main.py" (NOT WHEN IMPORTED)

if __name__ == "__main__":

    result = NH4inletModel4(param, inputData, flowThr)

    plot_model(inputData, result, start_date=start_date, end_date=end_date)
    plot_wet_day(inputData, result, flowThr)

    # ---------------------------------------
    # objective function
    wet = wet_mask(inputData, evalThr, lag_h)

    # OBJ A - NH4 concentration
    nse_all = calculate_nse(inputData["nh4_imputed"], result["simNH4conc"])
    nse_wet = calculate_nse(inputData.loc[wet, "nh4_imputed"], result.loc[wet, "simNH4conc"])

    # print("\nOBJ A - NH4 concentration")
    # print(f"NSE (all): {nse_all:.4f} | NSE (wet): {nse_wet:.4f} | 1-NSE: {1-nse_wet:.4f}")

    # OBJ B - NH4 event loads
    dt = inputData["time"].diff().dt.total_seconds().median() / 3600
    events = (wet & ~wet.shift(fill_value=False)).cumsum()

    # print("\nOBJ B - NH4 event loads")
    # for event in events[wet].unique():
    #     mask = wet & (events == event)
    #     obs, sim, err = calculate_event_load_error(
    #         inputData.loc[mask, "nh4_flux_smoothed"],
    #         result.loc[mask, "simNH4load"], dt
    #     )
    #     print(f"Event {event}: Obs = {obs:.1f} kg | Sim = {sim:.1f} kg | Error = {err:+.1f}%")