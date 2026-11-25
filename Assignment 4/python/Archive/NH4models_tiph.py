# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:43:38 2020

@author: luve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


data = pd.read_csv('data/data_preprocessed.csv', sep=";", parse_dates=['time'])
data['time'] = pd.to_datetime(data['time'], format='mixed', dayfirst=True)
data['date'] = data['time'].dt.date

def NH4inletModel0(param,inputData):
    
    
# ============================================================================
# MODEL #0 - Basic NH4 inlet model, based on the Fourier series
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#
#        - inputData: dataframe with flow data 

# ============================================================================

    # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    # calculate concentration
    Q=inputData['flow_smoothed'] #[m3/hr]
    NH4conc=DWFload/Q   #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel1(param,inputData):
    
    
# ============================================================================
# MODEL #1 - Extension of model0 with tank in series to account for volume placed
#            before sensors (e.g. primary clarifies)
# Inputs - Pars: array containing 
#              - the five parameters of the Fourier series according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#              - V = volume placed before the sensor
#        - inputData: dataframe with flow data 

# ============================================================================
    # reset index of the input dataframe, since Python weirdly keeps the old numbering
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
    # get the timestep of the time series
    dt=timeVector.to_series().diff().dt.total_seconds()/3600 # calculate the median difference between values (in case some data are missing)
    dt=np.nanmedian(dt)  # convert to hours (as float)
     
    # get flow data
    Q=inputData['flow_smoothed']

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    # ============  the upstream buffer volume ===============================
    noTanks=3 # number of conceptual CTSR used to represent advection in ups volume
    
    V=param[5]/noTanks # volume of each single conceputal tank
    noTimeStep=inputData.shape[0]
    M=np.empty([noTimeStep,noTanks]) #inizialize matrix with mass of each tank [g]
    DWFload_OutTank=np.empty([noTimeStep]) #inizialize matrix with mass of each tank [g]
    C0=DWFload[0]/Q[0] # [g/m3] assume an initial concentration, based on the first simulated value
    
    # use this concentration to initialize tanks
    for i in range(0,(noTanks)):
        M[0,i]=C0*V #[g]
    
    # Solve mass balance for the three tanks for all the simulation steps
    # general mass balance M[t]=M[t-1]+inFlux[t-1]-outFlux[t-1]

    for i in (range(1,noTimeStep)):
        # mass balance Tank #0 
        M[i,0]=M[i-1,0]+(DWFload[i-1] -        Q[i-1]*(M[i-1,0]/V))*dt #[g]
        # mass balance Tank #1
        M[i,1]=M[i-1,1]+(Q[i-1]*(M[i-1,0]/V) - Q[i-1]*(M[i-1,1]/V))*dt #[g]
        # mass balance Tank #2
        M[i,2]=M[i-1,2]+(Q[i-1]*(M[i-1,1]/V) - Q[i-1]*(M[i-1,2]/V))*dt #[g]
        # flux out of tanks
        DWFload_OutTank[i]=(M[i,noTanks-1]/V)*Q[i-1] #[g/hr]
    
    
    # calculate concentration
    NH4conc=DWFload_OutTank/Q  #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_OutTank
    out['simNH4conc']=NH4conc

    return out

def NH4inletModel2(param,inputData):
    
# ============================================================================
# MODEL #2 - Extension of model0 with an additional spike to describe steep 
#            rising in NH4 loads (tipically in the morning)
# Inputs - Pars: array containing 
#              - the eight parameters (five of of the Fourier series + three of the pulse)
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t) + 
#                           c1 * exp(-c2*(log10(t)-log10(c3))^2)
#
#
#        - inputData: dataframe with flow data 

# ============================================================================
    # reset index of the input dataframe, since Python weirdly keeps the old numbering
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
   
    # get flow data
    Q=inputData['flow']

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) + 
             param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY) +                  
             param[5]*np.exp(-param[6]*(np.log10(timeDOY)-np.log10(param[7]))**2))
    
    # calculate concentration
    NH4conc=DWFload/Q  #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel3(param,inputData,flowThr):
    
# ============================================================================
# MODEL #3 - Extension of model0 with a dilution model for when the event is ending 
#           similar to the restoration phase in Langeveld et al. (2017)
#            
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#               plus three additional parameters for defining the dilution (dF) in the restoration phase 
#               dF(tSEE)=c1*exp(-((log(tSEE)-c2)**2 )/c3))
#               with tSEE=time Since End of Event
#        - inputData: dataframe with flow data 
#        - flowThr: threshold for defining start and end of rain event
#
# ============================================================================
   # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    
    # =================  identify end of wet event =======================
    Q=inputData['flow_smoothed'] #[m3/hr]    
    idxFlowAboveThr=Q>flowThr #identify values when flow was above flow threshold
    
    # calculate the time passed since the last event
    timeStepSinceLastEvent=np.zeros(len(inputData))
    timeStepSinceLastEvent[0]=1440*365 #assume the start of the model in dry conditions (just put an unrealistic big value, such as 1 year)
    for i in range(1,len(inputData)): # loop over events
        if not idxFlowAboveThr[i]:
            timeStepSinceLastEvent[i]=timeStepSinceLastEvent[i-1]+1
            

    np.seterr(divide = 'ignore') #change settings - ignore error for log of zero
    dilutionFactor=(1-param[5]*np.exp(-((np.log(timeStepSinceLastEvent)-param[6])**2 )/param[7]))
    np.seterr(divide = 'warn') # re-activate error
    
    DWFload_new=DWFload*dilutionFactor
    
    
    # calculate concentration
    NH4conc=DWFload_new/Q   #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_new
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel4(param,inputData,flowThr):
    
# ============================================================================
# MODEL #4 - combination of Model 1 (tanks in series) annd model 3 (dilution)
#            
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#               plus three additional parameters for defining the dilution (dF) in the restoration phase 
#               dF(tSEE)=c1*exp(-((log(tSEE)-c2)**2 )/c3))
#               with tSEE=time Since End of Event
#               and V = volume placed before the sensor
#        - inputData: dataframe with flow data 
#        - flowThr: threshold for defining start and end of rain event
#
# ============================================================================
   # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
    # get the timestep of the time series
    dt=timeVector.to_series().diff().dt.total_seconds()/3600 # calculate the median difference between values (in case some data are missing)
    dt=np.nanmedian(dt)  # convert to hours (as float)
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    
    # =================  identify end of wet event =======================
    Q=inputData['flow_smoothed'] #[m3/hr]    
    idxFlowAboveThr=Q>flowThr #identify values when flow was above flow threshold
    
    # calculate the time passed since the last event
    timeStepSinceLastEvent=np.zeros(len(inputData))
    timeStepSinceLastEvent[0]=1440*365 #assume the start of the model in dry conditions (just put an unrealistic big value, such as 1 year)
    for i in range(1,len(inputData)): # loop over events
        if not idxFlowAboveThr[i]:
            timeStepSinceLastEvent[i]=timeStepSinceLastEvent[i-1]+1
            
    np.seterr(divide = 'ignore') #change settings - ignore error for log of zero
    dilutionFactor=(1-param[5]*np.exp(-((np.log(timeStepSinceLastEvent)-param[6])**2 )/param[7]))
    np.seterr(divide = 'warn') # re-activate error
    
    DWFload_new=DWFload*dilutionFactor
    
   # ============  the upstream buffer volume ===============================
    noTanks=3 # number of conceptual CTSR used to represent advection in ups volume
    
    V=param[8]/noTanks # volume of each single conceputal tank
    noTimeStep=inputData.shape[0]
    M=np.empty([noTimeStep,noTanks]) #inizialize matrix with mass of each tank [g]
    DWFload_OutTank=np.empty([noTimeStep]) #inizialize matrix with mass of each tank [g]
    C0=DWFload_new[0]/Q[0] # [g/m3] assume an initial concentration, based on the first simulated value
    
    # use this concentration to initialize tanks
    for i in range(0,(noTanks)):
        M[0,i]=C0*V #[g]
    
    # Solve mass balance for the three tanks for all the simulation steps
    # general mass balance M[t]=M[t-1]+inFlux[t-1]-outFlux[t-1]

    for i in (range(1,noTimeStep)):
        # mass balance Tank #0 
        M[i,0]=M[i-1,0]+(DWFload_new[i-1] -        Q[i-1]*(M[i-1,0]/V))*dt #[g]
        # mass balance Tank #1
        M[i,1]=M[i-1,1]+(Q[i-1]*(M[i-1,0]/V) - Q[i-1]*(M[i-1,1]/V))*dt #[g]
        # mass balance Tank #2
        M[i,2]=M[i-1,2]+(Q[i-1]*(M[i-1,1]/V) - Q[i-1]*(M[i-1,2]/V))*dt #[g]
        # flux out of tanks
        DWFload_OutTank[i]=(M[i,noTanks-1]/V)*Q[i-1] #[g/hr]
    
    
    # calculate concentration
    NH4conc=DWFload_OutTank/Q  #[g/m3]    
    
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_OutTank
    out['simNH4conc']=NH4conc

    return out



# def run_model0(param,inputData,measData,lb,ub):
    
#     import objective_function as objFun
    
        
#     # check for parameter bounds (since the simplex algorithm does not have bounds
#     # we need to give an absordly big value of the objective function to keep the
#     # algorithm inside the region of the parameter space we want to explore)
    

    
#     if all(lb<param)&all(param<ub): # the candidate point is within the  bounds
#         param=param*1e3
#         sim_out=NH4inletModel0(param,inputData)
        
#         RMSE=objFun.MARE(measData,sim_out['simNH4conc'])
#     else: # the candidate point is outside the  bounds
#         RMSE=1e300
        
#     return RMSE

# def run_model1(param,inputData,measData):
    
#     import objective_function as objFun
    
    
        
#     # check for parameter bounds (since the simplex algorithm does not have bounds
#     # we need to give an absordly big value of the objective function to keep the
#     # algorithm inside the region of the parameter space we want to explore)
    
#     param=param*1e3    
#     sim_out=NH4inletModel1(param,inputData)
        
#     RMSE=objFun.MARE(measData,sim_out['simNH4conc'])
        
#     # print("current obj fun value" ,str(RMSE) )
#     return RMSE


# def run_model1_BND(param,inputData,measData,lb,ub):
    
#     import objective_function as objFun
    
    
        
    # check for parameter bounds (since the simplex algorithm does not have bounds
    # we need to give an absordly big value of the objective function to keep the
    # algorithm inside the region of the parameter space we want to explore)
    
    if all(lb<param)&all(param<ub): # the candidate point is within the  bounds
        param=param*1e3    
        sim_out=NH4inletModel1(param,inputData)
        
        RMSE=objFun.MARE(measData,sim_out['simNH4conc'])
    else: # the candidate point is outside the  bounds
        RMSE=1e300
        
    # print("current obj fun value" ,str(RMSE) )
    return RMSE


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
ax1.plot(multi_data['time'], multi_data['sim_conc_m1'], color='blue', linestyle='-', linewidth=1.5, label='Model 1 (Tank)')
ax1.plot(multi_data['time'], multi_data['sim_conc_m4'], color='orange', linestyle='--', linewidth=1.5, label='Model 4 (Tank + Dilution)')
ax1.set_ylabel('NH4+ Flux [kg/h]', fontsize=11)
ax1.set_title(f'3-Day Calibration Window ({start_date.date()} to {(end_date - pd.Timedelta(days=1)).date()})', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add a parameter display box inside the plot so you can track your edits easily!
# Change this inside your text box definition:
param_text = f"Parameters Tested (in kg/h):\na0={param_m1[0]:.1f}, a1={param_m1[1]:.1f}, b1={param_m1[2]:.1f}, a2={param_m1[3]:.1f}, b2={param_m1[4]:.1f}"

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
plt.show()


# ==========================================
# 2. PLOT FOR THE WHOLE DATA TIME (2 Subplots)
# ==========================================
fig_all, (ax_all1, ax_all2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top Subplot: Full Time Series NH4+ Concentration
ax_all1.plot(data['time'], data['nh4_flux_smoothed']/ 1e3, color='black', alpha=0.6, linewidth=1, label='Measured (NH4+)')
ax_all1.plot(data['time'], data['sim_conc_m1'], color='blue', linewidth=1.2, label='Model 1 (Tank)')
ax_all1.plot(data['time'], data['sim_conc_m4'], color='orange', linewidth=1.2, label='Model 4 (Tank + Dilution)')
ax_all1.set_ylabel('NH4+ FLux [kg/h]', fontsize=11)
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
plt.show()
