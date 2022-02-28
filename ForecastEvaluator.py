import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_pinball_loss

def PredictionIntervalWidthCalculator(predictions, horizons):
    prediction_interval_widths = predictions

    prediction_interval_widths['IQR_25_75'] = predictions['0.75'] - predictions['0.25']
    prediction_interval_widths['IQR_025_975'] = predictions['0.975'] - predictions['0.025']

    avg_widths_25_75 = pd.DataFrame(horizons, columns=['horizon'])
    avg_widths_25_75['avg_width'] = np.zeros(len(avg_widths_25_75))
    avg_widths_025_975 = pd.DataFrame(horizons, columns=['horizon'])
    avg_widths_025_975['avg_width'] = np.zeros(len(avg_widths_025_975))

    for h in horizons:
        avg_widths_25_75['avg_width'][avg_widths_25_75['horizon'] == h] = prediction_interval_widths[prediction_interval_widths['horizon'] == h]['IQR_25_75'].mean()
        avg_widths_025_975['avg_width'][avg_widths_025_975['horizon'] == h] = prediction_interval_widths[prediction_interval_widths['horizon'] == h]['IQR_025_975'].mean()

    avg_widths_25_75_overall = avg_widths_25_75['avg_width'].mean()
    avg_widths_025_975_overall = avg_widths_025_975['avg_width'].mean()

    return prediction_interval_widths, avg_widths_25_75, avg_widths_025_975, avg_widths_25_75_overall, avg_widths_025_975_overall

def SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_gbm, prediction_interval_widths_qrf, str, target_variable):

    # data = [prediction_interval_widths_emos[str], prediction_interval_widths_emos_boosting[str], prediction_interval_widths_qrf[str]]
    # fig, ax = plt.subplots()
    # ax.set_title('Boxplots of IQR ' + str)
    # #ax = fig.add_axes([0, 0, 1, 1])
    # bp = ax.boxplot(data)
    # plt.show()
    #
    # data = {'emos': prediction_interval_widths_emos[str], 'emos+boosting': prediction_interval_widths_emos_boosting[str], 'qrf': prediction_interval_widths_qrf[str]}
    # fig, ax = plt.subplots()
    # ax.boxplot(data.values())
    # ax.set_xticklabels(data.keys())
    # ax.set_title('Boxplots of IQR ' + str + ' horizon ')
    # plt.show()

    data36 = {'emos': prediction_interval_widths_emos[prediction_interval_widths_emos['horizon'] == 36][str],
              'emos+boosting': prediction_interval_widths_emos_boosting[prediction_interval_widths_emos_boosting['horizon'] == 36][str],
              'qrf': prediction_interval_widths_qrf[prediction_interval_widths_qrf['horizon'] == 36][str],
              'gbm': prediction_interval_widths_gbm[prediction_interval_widths_gbm['horizon'] == 36][str]}
    data48 = {'emos': prediction_interval_widths_emos[prediction_interval_widths_emos['horizon'] == 48][str],
              'emos+boosting': prediction_interval_widths_emos_boosting[prediction_interval_widths_emos_boosting['horizon'] == 48][str],
              'qrf': prediction_interval_widths_qrf[prediction_interval_widths_qrf['horizon'] == 48][str],
              'gbm': prediction_interval_widths_gbm[prediction_interval_widths_gbm['horizon'] == 48][str]}
    data60 = {'emos': prediction_interval_widths_emos[prediction_interval_widths_emos['horizon'] == 60][str],
              'emos+boosting': prediction_interval_widths_emos_boosting[prediction_interval_widths_emos_boosting['horizon'] == 60][str],
              'qrf': prediction_interval_widths_qrf[prediction_interval_widths_qrf['horizon'] == 60][str],
              'gbm': prediction_interval_widths_gbm[prediction_interval_widths_gbm['horizon'] == 60][str]}
    data72 = {'emos': prediction_interval_widths_emos[prediction_interval_widths_emos['horizon'] == 72][str],
              'emos+boosting': prediction_interval_widths_emos_boosting[prediction_interval_widths_emos_boosting['horizon'] == 72][str],
              'qrf': prediction_interval_widths_qrf[prediction_interval_widths_qrf['horizon'] == 72][str],
              'gbm': prediction_interval_widths_gbm[prediction_interval_widths_gbm['horizon'] == 72][str]}
    data84 = {'emos': prediction_interval_widths_emos[prediction_interval_widths_emos['horizon'] == 84][str],
              'emos+boosting': prediction_interval_widths_emos_boosting[prediction_interval_widths_emos_boosting['horizon'] == 84][str],
              'qrf': prediction_interval_widths_qrf[prediction_interval_widths_qrf['horizon'] == 84][str],
              'gbm': prediction_interval_widths_gbm[prediction_interval_widths_gbm['horizon'] == 84][str]}

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(8,18), sharey=True)
    fig1.suptitle('Boxplots of ' + target_variable + ' ' + str)
    ax1.boxplot(data36.values())
    ax1.set_xticklabels(data36.keys())
    ax1.set_title('Boxplots of ' + target_variable + ' ' + str + ' horizon 36')
    ax2.boxplot(data48.values())
    ax2.set_xticklabels(data48.keys())
    ax2.set_title('Boxplots of ' + target_variable + ' ' + str + ' horizon 48')
    ax3.boxplot(data60.values())
    ax3.set_xticklabels(data60.keys())
    ax3.set_title('Boxplots of ' + target_variable + ' ' + str + ' horizon 60')
    ax4.boxplot(data72.values())
    ax4.set_xticklabels(data72.keys())
    ax4.set_title('Boxplots of ' + target_variable + ' ' + str + ' horizon 72')
    ax5.boxplot(data84.values())
    ax5.set_xticklabels(data84.keys())
    ax5.set_title('Boxplots of ' + target_variable + ' ' + str + ' horizon 84')
    if target_variable == "temperature":
        fig1.text(0.06, 0.5, 'temperature difference in degree celsius', ha='center', va='center', rotation='vertical')
    elif target_variable == "wind_speed":
        fig1.text(0.06, 0.5, 'wind speed in ', ha='center', va='center', rotation='vertical')
    plt.savefig('/Users/franziska/Dropbox/DataPTSFC/Plots/' + str + target_variable + 'boxplots.png')
    plt.show()
    return


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

quantile_preds_rw_emos_temp = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_7502022-01-11.csv')
quantile_preds_rw_emos_boosting_temp = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_7502022-01-11.csv')
quantile_preds_rw_gbm_temp = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_7502022-01-25.csv')
quantile_preds_rw_qrf_temp = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_7502022-01-25.csv')

winkler_score_emos_36_25_75 = winkler_score(quantile_preds_rw_emos_temp[quantile_preds_rw_emos_temp['horizon'] == 36]['obs'], quantile_preds_rw_emos_temp[quantile_preds_rw_emos_temp['horizon'] == 36]['0.25'], quantile_preds_rw_emos_temp[quantile_preds_rw_emos_temp['horizon'] == 36]['0.75'], 0.5)

quantile_preds_rw_emos_wind = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_750_wind_2022-01-12.csv')
quantile_preds_rw_emos_boosting_wind = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_750_wind_2022-01-12.csv')
quantile_preds_rw_gbm_wind = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_750_wind_2022-01-25.csv')
quantile_preds_rw_qrf_wind = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_750_wind_2022-01-25.csv')


horizons = [36, 48, 60, 72, 84]

"""
Calculate prediction interval widths and averages of them and save results in csv file 
"""
prediction_interval_widths_emos, avg_widths_25_75_emos, avg_widths_025_975_emos, avg_widths_25_75_overall_emos, avg_widths_025_975_overall_emos = PredictionIntervalWidthCalculator(quantile_preds_rw_emos_temp, horizons)
prediction_interval_widths_emos_boosting, avg_widths_25_75_emos_boosting, avg_widths_025_975_emos_boosting, avg_widths_25_75_overall_emos_boosting, avg_widths_025_975_overall_emos_boosting = PredictionIntervalWidthCalculator(quantile_preds_rw_emos_boosting_temp, horizons)
prediction_interval_widths_gbm, avg_widths_25_75_gbm, avg_widths_025_975_gbm, avg_widths_25_75_overall_gbm, avg_widths_025_975_overall_gbm = PredictionIntervalWidthCalculator(quantile_preds_rw_gbm_temp, horizons)
prediction_interval_widths_qrf, avg_widths_25_75_qrf, avg_widths_025_975_qrf, avg_widths_25_75_overall_qrf, avg_widths_025_975_overall_qrf = PredictionIntervalWidthCalculator(quantile_preds_rw_qrf_temp, horizons)

prediction_interval_widths_emos_wind, avg_widths_25_75_emos_wind, avg_widths_025_975_emos_wind, avg_widths_25_75_overall_emos_wind, avg_widths_025_975_overall_emos_wind = PredictionIntervalWidthCalculator(quantile_preds_rw_emos_wind, horizons)
prediction_interval_widths_emos_boosting_wind, avg_widths_25_75_emos_boosting_wind, avg_widths_025_975_emos_boosting_wind, avg_widths_25_75_overall_emos_boosting_wind, avg_widths_025_975_overall_emos_boosting_wind = PredictionIntervalWidthCalculator(quantile_preds_rw_emos_boosting_wind, horizons)
prediction_interval_widths_gbm_wind, avg_widths_25_75_gbm_wind, avg_widths_025_975_gbm_wind, avg_widths_25_75_overall_gbm_wind, avg_widths_025_975_overall_gbm_wind = PredictionIntervalWidthCalculator(quantile_preds_rw_gbm_wind, horizons)
prediction_interval_widths_qrf_wind, avg_widths_25_75_qrf_wind, avg_widths_025_975_qrf_wind, avg_widths_25_75_overall_qrf_wind, avg_widths_025_975_overall_qrf_wind = PredictionIntervalWidthCalculator(quantile_preds_rw_qrf_wind, horizons)

avg_widths_25_75 = avg_widths_25_75_emos
avg_widths_25_75 = avg_widths_25_75.rename(columns={'avg_width': 'avg_width_emos'})
avg_widths_25_75['avg_width_emos_boosting'] = avg_widths_25_75_emos_boosting['avg_width']
avg_widths_25_75['avg_width_gbm'] = avg_widths_25_75_gbm['avg_width']
avg_widths_25_75['avg_width_qrf'] = avg_widths_25_75_qrf['avg_width']

avg_widths_025_975 = avg_widths_025_975_emos
avg_widths_025_975 = avg_widths_025_975.rename(columns={'avg_width': 'avg_width_emos'})
avg_widths_025_975['avg_width_emos_boosting'] = avg_widths_025_975_emos_boosting['avg_width']
avg_widths_025_975['avg_width_gbm'] = avg_widths_025_975_gbm['avg_width']
avg_widths_025_975['avg_width_qrf'] = avg_widths_025_975_qrf['avg_width']

avg_widths_25_75.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_widths_25_75' + '.csv', index=False)
avg_widths_025_975.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_widths_025_975' + '.csv', index=False)

avg_widths_25_75_wind = avg_widths_25_75_emos_wind
avg_widths_25_75_wind = avg_widths_25_75_wind.rename(columns={'avg_width': 'avg_width_emos'})
avg_widths_25_75_wind['avg_width_emos_boosting'] = avg_widths_25_75_emos_boosting_wind['avg_width']
avg_widths_25_75_wind['avg_width_gbm'] = avg_widths_25_75_gbm_wind['avg_width']
avg_widths_25_75_wind['avg_width_qrf'] = avg_widths_25_75_qrf_wind['avg_width']

avg_widths_025_975_wind = avg_widths_025_975_emos_wind
avg_widths_025_975_wind = avg_widths_025_975_wind.rename(columns={'avg_width': 'avg_width_emos'})
avg_widths_025_975_wind['avg_width_emos_boosting'] = avg_widths_025_975_emos_boosting_wind['avg_width']
avg_widths_025_975_wind['avg_width_gbm'] = avg_widths_025_975_gbm_wind['avg_width']
avg_widths_025_975_wind['avg_width_qrf'] = avg_widths_025_975_qrf_wind['avg_width']

avg_widths_25_75_wind.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_widths_25_75_wind' + '.csv', index=False)
avg_widths_025_975_wind.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_widths_025_975_wind' + '.csv', index=False)

"""
Plot prediction interval widths in sharpness diagram
"""
SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_gbm, prediction_interval_widths_qrf, 'IQR_25_75', "temperature")
SharpnessDiagramPlotter(prediction_interval_widths_emos, prediction_interval_widths_emos_boosting, prediction_interval_widths_gbm, prediction_interval_widths_qrf, 'IQR_025_975', "temperature")
SharpnessDiagramPlotter(prediction_interval_widths_emos_wind, prediction_interval_widths_emos_boosting_wind, prediction_interval_widths_gbm_wind, prediction_interval_widths_qrf_wind, 'IQR_25_75', "wind")
SharpnessDiagramPlotter(prediction_interval_widths_emos_wind, prediction_interval_widths_emos_boosting_wind, prediction_interval_widths_gbm_wind, prediction_interval_widths_qrf_wind, 'IQR_025_975', "wind")

plt.figure(figsize=(17, 10))
plt.plot(quantile_preds_rw_emos_temp['obs_tm'][quantile_preds_rw_emos_temp['horizon'] == 36], quantile_preds_rw_emos_temp['0.025'][quantile_preds_rw_emos_temp['horizon'] == 36], label = '0.025')
plt.plot(quantile_preds_rw_emos_temp['obs_tm'][quantile_preds_rw_emos_temp['horizon'] == 36], quantile_preds_rw_emos_temp['0.975'][quantile_preds_rw_emos_temp['horizon'] == 36], label = '0.975')
plt.show()


plt.figure(figsize=(17, 10))
plt.plot(quantile_preds_rw_emos_temp['obs_tm'][quantile_preds_rw_emos_temp['horizon'] == 36], quantile_preds_rw_emos_temp['0.025'][quantile_preds_rw_emos_temp['horizon'] == 36], label = '0.025')
plt.plot(quantile_preds_rw_emos_temp['obs_tm'][quantile_preds_rw_emos_temp['horizon'] == 36], quantile_preds_rw_emos_temp['0.975'][quantile_preds_rw_emos_temp['horizon'] == 36], label = '0.975')
plt.show()

#prediction_interval_widths_emos.apply(lambda x: max(0.025 * (x['obs'][x['horizon'] == 36] - x['0.025'][x['horizon'] == 36]), (0.025 - 1) * (x['obs'][x['horizon'] == 36] - x['0.025'][x['horizon'] == 36])))


print(1)