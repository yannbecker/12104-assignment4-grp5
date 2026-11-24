import numpy as np

# meas, sim: arrays of same length, no nan
# optimiser minimises 1 - NSE

def NSE(meas, sim):
    # 1 = perfect, 0 = as good as the mean
    meas, sim = np.asarray(meas, float), np.asarray(sim, float)
    return 1 - np.sum((meas - sim) ** 2) / np.sum((meas - np.mean(meas)) ** 2)

def RMSE(meas, sim):
    meas, sim = np.asarray(meas, float), np.asarray(sim, float)
    return np.sqrt(np.mean((meas - sim) ** 2))

def MARE(meas, sim):
    meas, sim = np.asarray(meas, float), np.asarray(sim, float)
    return np.mean(np.abs((meas - sim) / meas))
