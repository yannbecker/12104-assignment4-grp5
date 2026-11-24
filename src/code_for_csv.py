from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # Assignment4/

# Loadin dataset : 
DATA_DIR = ROOT / 'dataAssignment4' / 'damhusaaen'
flow=pd.read_csv(DATA_DIR / 'flowdata_DAM_180901_181015.csv', sep=";", parse_dates=['time'])
nh4_concentration=pd.read_csv(DATA_DIR / 'NH4data_180901_181015.csv', sep=";", parse_dates=['time'])

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
data['flow_smoothed']=data['flow_imputed'].rolling(window=window_size, center=True, min_periods=1).mean()

# flag gaps > 60 min (30 steps), interpolated but not scored
MAX_GAP = 30
data['discard'] = False
for col in ['flow', 'nh4']:
    isna = data[col].isna()
    run_length = isna.groupby((isna != isna.shift()).cumsum()).transform('sum')
    data['discard'] |= isna & (run_length > MAX_GAP)

# wet / dry flags (dry = no exceedance in the last 24 h = 720 steps)
FLOW_THR = 5000
above = data['flow_smoothed'] > FLOW_THR
data['wet'] = above
data['dry'] = (above.rolling(720, min_periods=1).max() == 0) & ~data['discard']



#Calculating NH4+ flux :
data['nh4_flux'] = data['flow_imputed'] * data['nh4_imputed']
data['nh4_flux_smoothed'] = data['flow_smoothed'] * data['nh4_imputed']


#Export : 
output_filename = ROOT / 'combined_flow_nh4_flux.csv'
data.to_csv(output_filename, index=False, sep=';', date_format='%Y-%m-%d %H:%M:%S')   # iso dates

print(f"Successfully generated {output_filename} with {len(data)} rows.")


