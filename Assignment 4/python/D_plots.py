import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd



# ============================================================================
# MODEL - FULL DATASET (MEASURED VS. SIMULATED)
# ============================================================================

def plot_model(data, result, start_date=None, end_date=None):

    if start_date is not None:
        data = data[data["time"] >= pd.Timestamp(start_date)]
        result = result[result["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        data = data[data["time"] < pd.Timestamp(end_date)]
        result = result[result["time"] < pd.Timestamp(end_date)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True
    )
    ax1.plot(
        data["time"],
        data["nh4_imputed"],
        color="black",
        linewidth=1,
        alpha=0.7,
        label="Measured NH4 concentration"
    )
    ax1.plot(
        result["time"],
        result["simNH4conc"],
        color="blue",
        linewidth=1,
        label="Model 4 NH4 concentration"
    )
    ax1.set_ylabel("NH4 concentration [mg/L]")
    ax1.set_title("Model 4 - Measured vs. Simulated NH4")
    ax1.legend()
    ax1.grid(True, which="major", alpha=0.3)
    ax2.plot(
        data["time"],
        data["nh4_flux_smoothed"] / 1000,
        color="black",
        linewidth=1,
        alpha=0.7,
        label="Measured NH4 flux"
    )
    ax2.plot(
        result["time"],
        result["simNH4load"] / 1000,
        color="orange",
        linewidth=1,
        label="Model 4 NH4 flux"
    )
    ax2.set_ylabel("NH4 flux [kg/h]")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, which="major", alpha=0.3)
    ax2.xaxis.set_major_locator(
        mdates.DayLocator(interval=10)
    )
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m/%Y")
    )
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.savefig("output/full_data_measured_simulated_plot.pdf", dpi=300, bbox_inches="tight")
    plt.show()




# ============================================================================
# WET-DAY PLOT - MODEL 4
# ============================================================================

def plot_wet_day(data, result, flowThr):

    # Find the day with the highest measured flow
    data = data.copy()
    data["date"] = data["time"].dt.date
    daily_max_flow = data.groupby("date")["flow_imputed"].max()
    wettest_date = daily_max_flow.idxmax()
    print(f"Day with highest flow: {wettest_date}")

    # Select measurements and simulation for this day
    wet_day = data[data["date"] == wettest_date].copy()
    sim_wet = result[result["time"].dt.date == wettest_date].copy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True
    )
    ax1.plot(
        wet_day["time"],
        wet_day["nh4_imputed"],
        color="black",
        linewidth=1.5,
        label="Measured NH4"
    )
    ax1.plot(
        sim_wet["time"],
        sim_wet["simNH4conc"],
        color="orange",
        linestyle="--",
        linewidth=1.8,
        label="Model 4"
    )
    ax1.set_ylabel("NH4 concentration [mg/L]")
    ax1.set_title(f"High-Flow Event ({wettest_date})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(
        wet_day["time"],
        wet_day["flow_smoothed"],
        color="green",
        linewidth=1.5,
        label="Smoothed flow"
    )
    ax2.axhline(
        y=flowThr,
        color="red",
        linestyle=":",
        label=f"Flow threshold ({flowThr} m³/h)"
    )
    ax2.set_ylabel("Flow [m³/h]")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    plt.savefig("output/wetday_data_measured_simulated_plot.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================================
# PARAMETER TESTING - ONE AT A TIME
# ============================================================================

def plot_parameter_test(data, param, flowThr, param_index, test_values,
                        start_date=None, end_date=None):
    
    from B_Model import NH4inletModel4
    param_names = [
        "a0", "a1", "b1", "a2", "b2", "c1", "c2", "c3", "V", "flowThr"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True)
    
    # Time range for plotting
    mask = pd.Series(True, index=data.index)
    if start_date is not None:
        mask &= data["time"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= data["time"] < pd.Timestamp(end_date)
    
    # Measured data
    ax1.plot(
        data.loc[mask, "time"],
        data.loc[mask, "nh4_imputed"],
        color="black",
        linewidth=2,
        label="Measured NH4 concentration"
    )
    ax2.plot(
        data.loc[mask, "time"],
        data.loc[mask, "nh4_flux_smoothed"] / 1000,
        color="black",
        linewidth=2,
        label="Measured NH4 flux"
    )

    # Run model for each parameter value
    for value in test_values:
        p = param.copy()
        if param_index == 9:
            # Test flow threshold
            result = NH4inletModel4(p, data, value)
        else:
            # Test other parameters
            p[param_index] = value
            result = NH4inletModel4(p, data, flowThr)

        # Concentration
        ax1.plot(
            result.loc[mask, "time"],
            result.loc[mask, "simNH4conc"],
            linewidth=1.5,
            label=f"{param_names[param_index]} = {value:g}"
        )

        # Flux
        ax2.plot(
            result.loc[mask, "time"],
            result.loc[mask, "simNH4load"] / 1000,
            linewidth=1.5,
            label=f"{param_names[param_index]} = {value:g}"
        )

    # Formatting
    ax1.set_ylabel("NH4 concentration [mg/L]")
    ax2.set_ylabel("NH4 flux [kg/h]")
    ax2.set_xlabel("Time")

    ax1.set_title(
        f"Parameter sensitivity - {param_names[param_index]}"
    )
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("output/parameter_test_plot.pdf", dpi=300, bbox_inches="tight")
    plt.show()







# data = pd.read_csv("data/data_preprocessed.csv", sep=";", parse_dates=["time"], dayfirst=True)
# plt.figure(figsize=(12, 4))
# plt.plot(data["time"], data["flow_smoothed"], linewidth=1)
# plt.axhline(5000, color="red", linestyle="--", linewidth=1, label="5000 m³/h")
# plt.ylabel("Flow [m³/h]")
# plt.xlabel("Time")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()   