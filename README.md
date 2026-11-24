# 12104 Assignment 4 (Group 5)

Modelling NH4+ at the inlet of the Damhusåen wastewater treatment plant. We use **Model 4** (daily Fourier pattern + dilution in wet weather + buffer volume) and calibrate it on September to October 2018 data.

## What is in here

```
src/
  code_for_csv.py         merges raw flow + NH4 data, fills gaps, smooths flow, flags wet/dry -> combined_flow_nh4_flux.csv
  data_utils.py           loads the csv and splits it into calibration / validation
  NH4models.py            the course models (Models 0 to 4)
  objective_function.py   NSE, RMSE, MARE
  model_utils.py          helpers: run Model 4, evaluate, plot
  plausibility_check.py   Task 1.3: change parameters and see what the model does
  task3_calibration.py    Task 3: calibrate Model 4 with the optimiser you choose
  side_xgboost_benchmark.py  side experiment, not part of the report
dataAssignment4/          (to add) raw data from the course, see below
results/                  (created when you run the code) parameters and metrics as csv
figures/                  (created when you run the code) plots
```

## How to run it

1. Install the packages: `pip install numpy pandas scipy matplotlib`
2. Put the raw Damhusåen files in `dataAssignment4/damhusaaen/` (from learn).
3. Go to the `src` folder and run, in this order:

```
python code_for_csv.py                       # makes combined_flow_nh4_flux.csv
python plausibility_check.py                 # Task 1.3 plots
python task3_calibration.py Nelder-Mead      # Task 3 calibration
```

For the calibration you can swap the method name: `L-BFGS-B`, `differential_evolution`, `dual_annealing`.
Results go to `results/` and the fit plot to `figures/`, and the file names contain the method name.

## How the calibration works

- Step A: the 5 Fourier coefficients are fitted with Model 0 on **dry weather** only.
- Step B: with those fixed, the dilution parameters (c1, c2, c3) and the volume V are fitted on the whole calibration set.
- The objective is `1 - NSE`, and the first 24 h of each set are a warm-up that is not scored.
- Calibration is 1-27 Sept 2018, validation 28 Sept to 15 Oct (split in the longest dry gap).
- Parameter ranges are at the top of `task3_calibration.py`.

## Good to know

- The data/results/figures paths are relative to the folder above `src`, so keep that layout.
- Fourier parameters are in kg/h in the bounds and converted to g/h inside the code.

