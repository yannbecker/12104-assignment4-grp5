from D_plots import plot_parameter_test
from A_main import inputData, param, flowThr, start_date, end_date

param_index = 9  # Index of flowThr in the param array
test_values = [5000,11500,20000]
plot_parameter_test(
    inputData,
    param,
    flowThr,
    param_index,
    test_values,
    start_date=start_date,
    end_date=end_date
)


