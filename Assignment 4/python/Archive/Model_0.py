import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputData = pd.read_csv(
    'used_data/flowdata_DAM_180901_181015.csv',
    sep=';',
    parse_dates=['time']
)

param = np.array([
    125000,  # a0
    50000,   # a1
    50000,   # b1
    50000,   # a2
    50000    # b2
])

def NH4inletModel0(param,inputData):
# ============================================================================
# MODEL #0 - Basic NH4 inlet model, based on the Fourier series
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#        - inputData: dataframe with flow data 
# ============================================================================
    # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)
    # convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                           
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440  
    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY)) 
    # calculate concentration
    Q=inputData['flow'] #[m3/hr]
    NH4conc=DWFload/Q   #[g/m3]  
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload
    out['simNH4conc']=NH4conc
    return out

result = NH4inletModel0(param, inputData)
print(result.head())


fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(result['time'], result['simNH4load'] / 1000)
ax[0].set_ylabel('NH4 load [kg/h]')
ax[0].grid(True)
ax[1].plot(result['time'], result['simNH4conc'])
ax[1].set_ylabel('NH4 concentration [mg/L]')
ax[1].set_xlabel('Time')
ax[1].grid(True)
plt.tight_layout()
plt.show()