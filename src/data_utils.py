"""load data and split cal/val"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # Assignment4/
RESULTS_DIR = ROOT / "results"                  # csv / npy outputs
RESULTS_DIR.mkdir(exist_ok=True)

FLOW_THR = 5000            # Damhusaaen wet-weather threshold in m^3/hr
# split in the longest dry gap (24 Sep 23:00 -> 2 Oct 14:30), no event cut
# cal: 11 Sep, 21-22 Sep, 24 Sep | val: 2-3 Oct, 7 Oct
SPLIT_TIME = pd.Timestamp("2018-09-28 00:00")
WARMUP_H = 24              # [h] not scored at the start of each set


def load_and_split(split_time=SPLIT_TIME, verbose=True):
    # returns cal, val, cal_dry
    df = pd.read_csv(ROOT / "combined_flow_nh4_flux.csv", sep=";", parse_dates=["time"])   # made by code_for_csv.py
    df["flow"], df["nh4"] = df["flow_smoothed"], df["nh4_imputed"]   # NH4models.py reads flow_smoothed
    # model 4 cannot take nan flow: trim nan edges, interpolated gaps run but are not scored (valid)
    df = df.dropna(subset=["flow"]).reset_index(drop=True)
    df["valid"] = ~df["discard"] & df["nh4"].notna()

    if split_time is None:
        split_time = df["time"].iloc[len(df) // 2]
    cal = df[df["time"] < split_time].reset_index(drop=True)
    val = df[df["time"] >= split_time].reset_index(drop=True)
    cal_dry = cal[cal["dry"] & cal["valid"]].reset_index(drop=True)   # dry only (step A)

    # warm-up: model 4 restarts at the start of each set, first hours run but are not scored
    for d in (cal, val):
        d["warmup"] = d["time"] < d["time"].iloc[0] + pd.Timedelta(hours=WARMUP_H)
        d["scored"] = d["valid"] & ~d["warmup"]

    if verbose:   # both sets need wet events
        for name, d in [("calibration", cal), ("validation", val)]:
            ev_start = ((d["flow"] > FLOW_THR) & ~(d["flow"] > FLOW_THR).shift(fill_value=False)).sum()
            print(f"{name:12s}: {d['time'].iloc[0]} -> {d['time'].iloc[-1]}, "
                  f"{len(d)} steps, {ev_start} wet event(s), {100 * d['wet'].mean():.0f} % wet")
    return cal, val, cal_dry
