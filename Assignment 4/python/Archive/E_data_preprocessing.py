import pandas as pd
import matplotlib.pyplot as plt

# Loadin dataset : 
flow=pd.read_csv('data/flowdata_DAM_180901_181015.csv', sep=";", parse_dates=['time'])
nh4_concentration=pd.read_csv('data/NH4data_180901_181015.csv', sep=";", parse_dates=['time'])

#clean column names : 
flow.columns = [c.replace('"', '') for c in flow.columns]
nh4_concentration.columns = [c.replace('"', '') for c in nh4_concentration.columns]

# put data in new csv alltogether :
data = pd.merge(flow, nh4_concentration, on='time', how='outer').sort_values('time').reset_index(drop=True)

# data imputation : Linear interpolation + forward/backward fill for any boundary NaNs
data['flow_imputed'] = data['flow'].interpolate(method='linear').ffill().bfill()
data['nh4_imputed'] = data['nh4'].interpolate(method='linear').ffill().bfill()

# Smooth flow data with moving average (30 min centered moving average) :
window_size = 15
data['flow_smoothed']=data['flow_imputed'].rolling(window=window_size).mean().bfill().ffill()

#Calculating NH4+ flux :
data['nh4_flux'] = data['flow_imputed'] * data['nh4_imputed']
data['nh4_flux_smoothed'] = data['flow_smoothed'] * data['nh4_imputed']

#Export : 
output_filename = 'combined_flow_nh4_flux.csv'
# data.to_csv(output_filename, index=False, sep=';')

print(f"Successfully generated {output_filename} with {len(data)} rows.")

fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)


# plot
# 1. Flow
axs[0].plot(data['time'], data['flow'], label='Original', alpha=0.5)
axs[0].plot(data['time'], data['flow_imputed'], label='Imputed')
axs[0].plot(data['time'], data['flow_smoothed'], label='Smoothed')
axs[0].set_ylabel('Flow')
axs[0].legend()
axs[0].grid(True)

# 2. NH4 concentration
axs[1].plot(data['time'], data['nh4'], label='Original', alpha=0.5)
axs[1].plot(data['time'], data['nh4_imputed'], label='Imputed')
axs[1].set_ylabel('NH4 concentration')
axs[1].legend()
axs[1].grid(True)

# 3. NH4 flux
axs[2].plot(data['time'], data['nh4_flux'], label='NH4 flux')
axs[2].plot(data['time'], data['nh4_flux_smoothed'],
            label='NH4 flux (smoothed flow)')
axs[2].set_ylabel('NH4 flux')
axs[2].legend()
axs[2].grid(True)

# 4. Comparison of flow processing
axs[3].plot(data['time'], data['flow_imputed'], label='Imputed', alpha=0.5)
axs[3].plot(data['time'], data['flow_smoothed'], label='Smoothed')
axs[3].set_ylabel('Flow')
axs[3].set_xlabel('Time')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
# plt.savefig('flow_nh4_analysis.pdf', bbox_inches='tight')
plt.show()
