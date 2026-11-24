"""Plausibility check of Models 1 and 4"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from NH4models import NH4inletModel1, NH4inletModel4
from data_utils import ROOT

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

data = pd.read_csv(ROOT / 'combined_flow_nh4_flux.csv', sep=";", parse_dates=['time'])
data['date'] = data['time'].dt.date

# ==========================================
# RUN SIMULATIONS
# ==========================================
# Example parameters (adjust these based on your calibrated optimization results)
# Model 1 parameters: [a0, a1, b1, a2, b2, V] (scaled by 1e3 as in your optimization function)

fourier_p = np.array([124.6, -32.3, -15.6, -5.88, 17.4]) * 1e3
dilution_p = np.array([0.45, 0.5, 8])  # c1, c2, c3
V_p = 11500                              # Volume

param_m1 = np.concatenate([fourier_p, [V_p]])
# Model 4 parameters: [a0, a1, b1, a2, b2, c1, c2, c3, V]
param_m4 = np.concatenate([fourier_p, dilution_p, [V_p]])

flow_threshold = 5000  # from table

sim1 = NH4inletModel1(param_m1, data)
sim4 = NH4inletModel4(param_m4, data, flow_threshold)

data['sim_conc_m1'] = sim1['simNH4conc']
data['sim_conc_m4'] = sim4['simNH4conc']
data['sim_load_m1'] = sim1['simNH4load'] / 1e3   # kg/h
data['sim_load_m4'] = sim4['simNH4load'] / 1e3



# ==========================================
# 1. PLOT FOR 3 CONSECUTIVE DAYS (2 Subplots + Parameter Box)
# ==========================================
start_date = pd.to_datetime('2018-09-10')
end_date = pd.to_datetime('2018-09-13') # 3-day window (Sep 10 to Sep 12)

multi_data = data[(data['time'] >= start_date) & (data['time'] < end_date)].copy()
print(f"Plotting 3-day window from {start_date.date()} to {end_date.date()}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top Subplot: NH4+ Concentration Comparison
ax1.plot(multi_data['time'], multi_data['nh4_flux_smoothed']/ 1e3, color='black', linewidth=1.5, label='Measured (NH4+)')
ax1.plot(multi_data['time'], multi_data['sim_load_m1'], color='blue', linestyle='-', linewidth=1.5, label='Model 1 (Tank)')
ax1.plot(multi_data['time'], multi_data['sim_load_m4'], color='orange', linestyle='--', linewidth=1.5, label='Model 4 (Tank + Dilution)')
ax1.set_ylabel('NH4+ Flux [kg/h]', fontsize=11)
ax1.set_title(f'3-Day Calibration Window ({start_date.date()} to {(end_date - pd.Timedelta(days=1)).date()})', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add a parameter display box inside the plot so you can track your edits easily!
# Change this inside your text box definition:
p_kg = param_m1[:5] / 1e3   # g/h to kg/h
param_text = (f"Parameters Tested (in kg/h):\na0={p_kg[0]:.1f}, a1={p_kg[1]:.1f}, b1={p_kg[2]:.1f}, "
              f"a2={p_kg[3]:.1f}, b2={p_kg[4]:.1f}")

ax1.text(0.02, 0.05, param_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Bottom Subplot: Smoothed Flow Comparison
ax2.plot(multi_data['time'], multi_data['flow_smoothed'], color='green', linewidth=1.5, label='Smoothed Flow')
ax2.set_ylabel('Flow [m3/hr]', fontsize=11)
ax2.set_xlabel('Date & Time', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Format X-axis for multi-day hours/days
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=20)

fig.tight_layout()
fig.savefig(FIG_DIR / "plausibility_1_three_days.png", dpi=200)
plt.show()


# ==========================================
# 2. PLOT FOR THE WHOLE DATA TIME (2 Subplots)
# ==========================================
fig_all, (ax_all1, ax_all2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top Subplot: Full Time Series NH4+ Concentration
ax_all1.plot(data['time'], data['nh4_flux_smoothed']/ 1e3, color='black', alpha=0.6, linewidth=1, label='Measured (NH4+)')
ax_all1.plot(data['time'], data['sim_load_m1'], color='blue', linewidth=1.2, label='Model 1 (Tank)')
ax_all1.plot(data['time'], data['sim_load_m4'], color='orange', linewidth=1.2, label='Model 4 (Tank + Dilution)')
ax_all1.set_ylabel('NH4+ Flux [kg/h]', fontsize=11)
ax_all1.set_title('Full Dataset Time Series: Concentration & Flow Comparison', fontsize=13, fontweight='bold')
ax_all1.legend(loc='upper right')
ax_all1.grid(True, alpha=0.3)

# Bottom Subplot: Full Time Series Smoothed Flow
ax_all2.plot(data['time'], data['flow_smoothed'], color='green', linewidth=1.2, label='Smoothed Flow')
ax_all2.set_ylabel('Flow [m3/hr]', fontsize=11)
ax_all2.set_xlabel('Date', fontsize=11)
ax_all2.legend(loc='upper right')
ax_all2.grid(True, alpha=0.3)

# Format X-axis for days
ax_all2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax_all2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30)

fig_all.tight_layout()
fig_all.savefig(FIG_DIR / "plausibility_2_whole_period.png", dpi=200)
plt.show()


daily_max_flow = data.groupby('date')['flow_imputed'].max()
wettest_date = daily_max_flow.idxmax()
print(f"The wettest day in the dataset is: {wettest_date}")

wet_day_data = data[data['date'] == wettest_date].copy()

# ==========================================
# PLOT FOR THE WETTEST DAY (2 Subplots)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top Subplot: NH4+ Concentration Comparison (Model 4 should now diverge from Model 1!)
ax1.plot(wet_day_data['time'], wet_day_data['nh4_imputed'], color='black', linewidth=1.5, label='Measured (NH4+)')
ax1.plot(wet_day_data['time'], wet_day_data['sim_conc_m1'], color='blue', linestyle='-', linewidth=1.5, label='Model 1 (Tank)')
ax1.plot(wet_day_data['time'], wet_day_data['sim_conc_m4'], color='orange', linestyle='--', linewidth=1.8, label='Model 4 (Tank + Dilution)')
ax1.set_ylabel('NH4+ Conc. [mg/L]', fontsize=11)
ax1.set_title(f'Wet-Weather Event Comparison on Wettest Day ({wettest_date})', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom Subplot: Smoothed Flow (Showing the rain peak crossing the 5000 m3/h threshold)
ax2.plot(wet_day_data['time'], wet_day_data['flow_smoothed'], color='green', linewidth=1.5, label='Smoothed Flow')
ax2.axhline(y=flow_threshold, color='red', linestyle=':', label=f'Flow Threshold ({flow_threshold} m3/h)')
ax2.set_ylabel('Flow [m3/hr]', fontsize=11)
ax2.set_xlabel('Time of Day (Hour:Minute)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Format X-axis for hours
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig.tight_layout()
fig.savefig(FIG_DIR / "plausibility_3_wettest_day.png", dpi=200)
plt.show()
