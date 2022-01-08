import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def PredictionIntervalWidthCalculator(predictions, horizons):

    prediction_interval_widths = pd.DataFrame(np.zeros((len(predictions), 2)), columns=['IQR_25_75', 'IQR_025_975'])
    prediction_interval_widths['IQR_25_75'] = predictions['0.75'] - predictions['0.25']
    prediction_interval_widths['IQR_025_975'] = predictions['0.975'] - predictions['0.025']

    avg_widths_25_75 = pd.DataFrame(horizons, columns=['horizon'])
    avg_widths_25_75['avg_width'] = np.zeros(len(avg_widths_25_75))
    avg_widths_025_975 = pd.DataFrame(horizons, columns=['horizon'])
    avg_widths_025_975['avg_width'] = np.zeros(len(avg_widths_025_975))

    for h in horizons:
        avg_widths_25_75['avg_width'][avg_widths_25_75['horizon'] == h] = prediction_interval_widths[prediction_interval_widths['horizon'] == h]['IQR_25_75'].mean()
        avg_widths_025_975['avg_width'][avg_widths_025_975['horizon'] == h] = prediction_interval_widths[prediction_interval_widths['horizon'] == h]['IQR_025_975'].mean()

    avg_widths_25_75_overall = avg_widths_25_75.mean()
    avg_widths_025_975_overall = avg_widths_025_975.mean()

    return prediction_interval_widths, avg_widths_25_75, avg_widths_025_975, avg_widths_25_75_overall, avg_widths_025_975_overall

def SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_qrf, str):

    data = [prediction_interval_widths_emos[str], prediction_interval_widths_emos_boosting[str], prediction_interval_widths_qrf[str]]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)
    plt.show()

    return

#SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_qrf, 'IQR_25_75')
#SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_qrf, 'IQR_025_975')


def QuantilePredictionEvaluator(predictions, quantile_levels, horizons):
    avg_pinball_loss = pd.DataFrame(horizons, columns=['horizon'])
    avg_pinball_loss[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(avg_pinball_loss))

    for q in quantile_levels:
        for h in horizons:
            avg_pinball_loss[str(q)][avg_pinball_loss['horizon'] == h] = mean_pinball_loss(predictions['obs'][predictions['horizon'] == h], predictions[str(q)][predictions['horizon'] == h], alpha=q)

    avg_pinball_loss['avg_per_horizon'] = avg_pinball_loss[['0.025', '0.25', '0.5', '0.75', '0.975']].mean(axis = 1)
    avg_pinball_loss_per_quantile = avg_pinball_loss[['0.025', '0.25', '0.5', '0.75', '0.975']].mean(axis = 0)
    avg_pinball_loss_overall = avg_pinball_loss_per_quantile.mean()
    return avg_pinball_loss, avg_pinball_loss_per_quantile, avg_pinball_loss_overall


def winkler_score(data, PI_LB, PI_UB, alpha):
    n = len(data)
    winkler_score_vec = np.zeros((n, 1))
    for i in range(0, n):
        delta_t = PI_UB[i] - PI_LB[i]
        if PI_LB[i] <= data[i] <= PI_UB[i]:
            winkler_score_vec[i] = delta_t
        elif data[i] < PI_LB[i]:
            delta_t + 2 / alpha * (PI_LB[i] - data[i])
        else:
            delta_t + 2 / alpha * (data[i] - PI_UB[i])

    return np.mean(winkler_score_vec)


#winkler_score_var_histsim = winkler_score(data_1718.fe, PIs_var.histsim_l, PIs_var.histsim_u, 0.05)
