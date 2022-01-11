import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from datetime import datetime, timedelta
from Dataloader_weather import DataUpdaterWeather
from WeatherForecastModels import EMOS_wind, GBM, QRF, RollingWindowQuantileCalculator, QRAFitterAndEvaluator, QRAQuantilePredictor
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss

"""
In the following we first set up the rpy2 framework in order to be able to use R packages afterwards
"""
base = importr('base')
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# R package names which are required in the following and therefore now installed first
packnames = ('ggplot2', 'hexbin', 'scoringRules', 'rdwd')
from rpy2.robjects.vectors import StrVector

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

xfun = importr('xfun')
scoringRules = rpackages.importr('scoringRules')
crch = rpackages.importr('crch')
""" load wind data """
#weather_data = DataUpdaterWeather(datetime.strftime(datetime.now(), '%Y-%m-%d'))
#weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather('2021-12-15', 'wind')
weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather(datetime.strftime(datetime.now(), '%Y-%m-%d'), 'wind')
"""
First visualize real wind observations to get a feeling for the data
"""
logging.info('Starting visualization of wind data')

df_wind_10m_plot = df_wind_10m.dropna()
ind = 1
for year in df_wind_10m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year).unique():
    plt.plot(df_wind_10m_plot['obs_tm'][df_wind_10m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year) == year],
             df_wind_10m_plot['obs'][df_wind_10m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year) == year])
    plt.xlabel('time')
    plt.ylabel('wind in km/h')
    ind = ind + 1
    plt.show()
    plt.savefig('/Users/franziska/Dropbox/DataPTSFC/Plots/' + str(year) + 'wind_timeseries_raw_data.png')

horizon = [36, 48, 60, 72, 84]
estimated_params = pd.DataFrame(horizon, columns=['horizon'])
estimated_params['mu'] = np.zeros(len(estimated_params))
estimated_params['sd'] = np.zeros(len(estimated_params))
estimated_params['crps'] = np.zeros(len(estimated_params))
estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))
for i in horizon:
    wind_10m_data_fcsth_i = df_wind_10m[(df_wind_10m['fcst_hour'] == i)]
    #wind_10m_data_fcsth_i = wind_10m_data_fcsth_i[wind_10m_data_fcsth_i['init_tm'].dt.month.isin([10, 11, 12])]
    wind_10m_data_fcsth_i_train = wind_10m_data_fcsth_i[['ens_mean', 'ens_sd', 'obs']].iloc[0:len(wind_10m_data_fcsth_i) - 1]
    wind_10m_data_fcsth_i_test = wind_10m_data_fcsth_i[['ens_mean', 'ens_sd']].iloc[-1:]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        wind_10m_data_fcsth_i_train_r = robjects.conversion.py2rpy(wind_10m_data_fcsth_i_train)
        wind_10m_data_fcsth_i_test_r = robjects.conversion.py2rpy(wind_10m_data_fcsth_i_test)

    pandas2ri.activate()
    robjects.globalenv['wind_10m_data_fcsth_i_train_r'] = wind_10m_data_fcsth_i_train

    robjects.r('''
                f <- function(wind_10m_data_fcsth_i_train) {

                        library(crch)
                        train1.crch <- crch(obs ~ ens_mean|ens_sd, data = wind_10m_data_fcsth_i_train_r, dist = "gaussian", left = 0, truncated = TRUE, type = "crps", link.scale = "log")

                }
                ''')
    r_f = robjects.globalenv['f']
    rf_model = (r_f(wind_10m_data_fcsth_i_train_r))
    res = pandas2ri.rpy2py(rf_model)
    robjects.r('''
               g <- function(model,test) {

                        pred_loc <- as.data.frame(predict(model, test, type = "location"))

                }

                h <- function(model,test) {

                        pred_loc <- as.data.frame(predict(model, test, type = "scale"))

                }
                ''')

    r_g = robjects.globalenv['g']
    r_h = robjects.globalenv['h']
    prediction_mu = (r_g(rf_model, wind_10m_data_fcsth_i_test_r)).values
    prediction_sd = (r_h(rf_model, wind_10m_data_fcsth_i_test_r)).values
    # crps_fun = scoringRules.crps
    # r_float = robjects.vectors.FloatVector
    # y_true_r = r_float(wind_10m_data_fcsth_i_test['obs'])
    # mu_r = r_float(prediction_mu)
    # sigma_r = r_float(prediction_sd)
    # score = scoringRules.crps(y_true_r, mean=mu_r, sd=sigma_r, family="normal")
    #    mean_crps_score = np.array(score).mean()

    estimated_params['mu'][estimated_params['horizon'] == i] = prediction_mu
    estimated_params['sd'][estimated_params['horizon'] == i] = prediction_sd
#    estimated_params['crps'][estimated_params['horizon'] == i] = score

    quantile_levels = [0.025,0.25,0.5,0.75,0.975]

    for q in quantile_levels:
        percentile_q = norm(loc=estimated_params['mu'][estimated_params['horizon'] == i], scale=estimated_params['sd'][estimated_params['horizon'] == i]).ppf(q)
        estimated_params[str(q)][estimated_params['horizon'] == i] = percentile_q
    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)

#estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

""" EMOS and boosting """
estimated_params_boost_EMOS = pd.DataFrame(horizon, columns=['horizon'])
estimated_params_boost_EMOS['mu'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS['sd'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params_boost_EMOS))

for i in horizon:
    wind_10m_data_fcsth_i = df_wind_10m[(df_wind_10m['fcst_hour'] == i)]
    wind_10m_data_fcsth_i = wind_10m_data_fcsth_i[wind_10m_data_fcsth_i['init_tm'].apply(lambda x: x.to_pydatetime().month).isin([11, 12, 1])]
    #wind_10m_data_fcsth_i = wind_10m_data_fcsth_i[wind_10m_data_fcsth_i['init_tm'].dt.month.isin([10, 11, 12])]

    wind_10m_data_fcsth_i_train = wind_10m_data_fcsth_i[['ens_mean', 'ens_sd', 'obs']].iloc[0:len(wind_10m_data_fcsth_i) - 1]
    wind_10m_data_fcsth_i_test = wind_10m_data_fcsth_i[['ens_mean', 'ens_sd']].iloc[-1:]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        wind_10m_data_fcsth_i_train_r = robjects.conversion.py2rpy(wind_10m_data_fcsth_i_train)
        wind_10m_data_fcsth_i_test_r = robjects.conversion.py2rpy(wind_10m_data_fcsth_i_test)

    pandas2ri.activate()
    robjects.globalenv['wind_10m_data_fcsth_i_train_r'] = wind_10m_data_fcsth_i_train
    robjects.r('''
               f <- function(wind_10m_data_fcsth_i_train) {

                        library(crch)
                        train1.crch <- crch(obs ~ ens_mean|ens_sd, data = wind_10m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log", left = 0, truncated = TRUE, control = crch.boost(mstop = "aic"))

                }
                ''')

    r_f = robjects.globalenv['f']
    rf_model = (r_f(wind_10m_data_fcsth_i_train_r))
    res = pandas2ri.rpy2py(rf_model)
    robjects.r('''
               g <- function(model,test) {

                        pred_loc <- as.data.frame(predict(model, test, type = "location"))

                }

                h <- function(model,test) {

                        pred_loc <- as.data.frame(predict(model, test, type = "scale"))

                }
                ''')

    r_g = robjects.globalenv['g']
    r_h = robjects.globalenv['h']
    prediction_mu = (r_g(rf_model, wind_10m_data_fcsth_i_test_r)).values
    prediction_sd = (r_h(rf_model, wind_10m_data_fcsth_i_test_r)).values

    estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i] = prediction_mu
    estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i] = prediction_sd
#    estimated_params['crps'][estimated_params['horizon'] == i] = score

    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        #percentile_q = truncnorm.ppf(q = q, a = 0, b = 50, loc=estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i], scale=estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i])
        percentile_q = norm(loc=estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i], scale=estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i]).ppf(q)

        estimated_params_boost_EMOS[str(q)][estimated_params_boost_EMOS['horizon'] == i] = percentile_q

    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
estimated_params_boost_EMOS['0.025'][estimated_params_boost_EMOS['horizon'] == 72] = estimated_params['0.025'][estimated_params['horizon'] == 72]
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

ind_boosting = True
for i in horizon:
    weather_data_i = weather_data[(weather_data['fcst_hour'] == i)].reset_index().drop(columns = 'index')
    weather_data_i = weather_data_i.drop(columns = ['init_tm', 'obs_tm'])
    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        estimated_params[str(q)][estimated_params['horizon'] == i] = EMOS_wind(q, weather_data_i.drop(columns='obs').iloc[0:len(weather_data_i) - 1], weather_data_i['obs'].iloc[0:len(weather_data_i) - 1], weather_data_i.drop(columns='obs').iloc[-1:], ind_boosting)

    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

print(1)
"""
Relative evaluation of the different models with each other to check which model performs best in terms of evaluation criterion
Use average over linear quantile scores since that's an approximation of the CRPS which is a strictly proper scoring rule
"""

# def RollingWindowQuantileCalculator(model, data, length_train_data, index_drop_na, horizon, emos_ind_boosting, considered_days):
#
#     ind = 1
#
#     for h in horizon:
#         data_h = data[(data['fcst_hour'] == h)]
#         if index_drop_na == True:
#             data_h = data_h.dropna()
#
#         data_h = data_h.reset_index()
#         data_h = data_h.drop(columns='index')
#         len_data = len(data_h)
#         len_preds = len_data - length_train_data
#
#         # Dataframe that contains quantile predictions for the different horizons and test data times
#         quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[length_train_data:len_data-1], columns=['init_tm','obs_tm'])
#         quantile_preds_rw_h['horizon'] = h
#         quantile_preds_rw_h[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len_preds-1, 5))
#
#         quantile_preds_rw_h = quantile_preds_rw_h.reset_index()
#         quantile_preds_rw_h = quantile_preds_rw_h.drop(columns='index')
#
#         for i in range(0, len_preds - 1):
#
#             X_y_train = data_h.iloc[i:i+length_train_data]
#             X_y_train = X_y_train[(X_y_train['init_tm'].apply(lambda x: x.to_pydatetime().month) <= (
#                         data_h['init_tm'].iloc[i + length_train_data] + timedelta(
#                         days=considered_days)).to_pydatetime().month) & (
#                         X_y_train['init_tm'].apply(lambda x: x.to_pydatetime().month) >= (
#                         data_h['init_tm'].iloc[i + length_train_data] + timedelta(
#                         days=considered_days)).to_pydatetime().month)]
#             X_train = X_y_train.drop(columns='obs')
#             y_train = X_y_train['obs']
#             time = data_h['init_tm'].iloc[i + length_train_data]
#             X_test = data_h.drop(columns=['obs', 'init_tm', 'fcst_hour', 'obs_tm']).iloc[i + length_train_data]
#
#             for q in quantile_levels:
#                 if model == EMOS_wind:
#                     quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
#                         columns=['init_tm', 'obs_tm']), y_train, X_test.drop(
#                         columns=['init_tm', 'obs_tm']).to_frame().T, emos_ind_boosting)
#
#                 else:
#                     quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
#                         columns=['init_tm', 'fcst_hour', 'obs_tm']), y_train, X_test)
#
#         if ind == 1:
#             quantile_preds_rw = quantile_preds_rw_h
#         else:
#             quantile_preds_rw = quantile_preds_rw.append(quantile_preds_rw_h)
#
#         ind = ind + 1
#
#     return quantile_preds_rw


quantile_preds_rw_emos_boosting_submission = RollingWindowQuantileCalculator(EMOS_wind, weather_data, len(weather_data)-1, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366)
quantile_preds_rw_emos_boosting_submission[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)


quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS_wind, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS_wind, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366)
quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)


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

quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)

quantile_preds_rw_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
avg_pinball_loss_emos_boosting, avg_pinball_loss_per_quantile_emos_boosting, avg_pinball_loss_overall_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)

quantile_preds_rw_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_boosting_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_boosting_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)

quantile_preds_rw_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_gbm_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_gbm_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)

quantile_preds_rw_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qrf_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qrf_600_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

#
"""
Grid search of parameters with rolling window evaluation 
"""
# TEST IF FORECAST PERFORMANCE IMPROVES IF ONLY DATA FROM MONTHS SIMILAR TO MONTH FOR WHICH CURRENT QUANTILES ARE
# PREDICTED IS USED FOR MODEL TRAINING BECAUSE BIASES AND VARIABILITY COULD DEPEND ON TIME OF YEAR
# Note: of course this causes different training data sizes based on date that's predicted but since it's consistently
# done for all models we can compare them
# Alternative would be shorter training data sizes because then we only choose last few days/months but we wouldn't choose from months that come after month of current prediction

array_len_days = [92, 155, 220, 366]
average_qs_len_months = pd.DataFrame(array_len_days, columns=['nbr_considered_months'])
average_qs_len_months[['EMOS', 'EMOS_boosting', 'Boosting', 'QRF']] = np.zeros((len(average_qs_len_months), 4))
for m in array_len_days:

    quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS_wind, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m)
    quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS_wind, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = m)
    quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m)
    quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 600, index_drop_na=True, horizon=horizon,
                                                            emos_ind_boosting=False, considered_days=m)

    quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
    avg_pinball_loss_emos_d, avg_pinball_loss_per_quantile_emos_d, avg_pinball_loss_overall_emos_d = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
    average_qs_len_months['EMOS'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_emos_d

    quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
    avg_pinball_loss_emos_boosting_d, avg_pinball_loss_per_quantile_emos_boosting_d, avg_pinball_loss_overall_emos_boosting_d = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
    average_qs_len_months['EMOS_boosting'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_emos_boosting_d

    quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    avg_pinball_loss_gbm_d, avg_pinball_loss_per_quantile_gbm_d, avg_pinball_loss_overall_gbm_d = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
    average_qs_len_months['Boosting'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_gbm_d

    quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    avg_pinball_loss_qrf_d, avg_pinball_loss_per_quantile_qrf_d, avg_pinball_loss_overall_qrf_d = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
    average_qs_len_months['QRF'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_qrf_d


eval_data_all_horizons = QRAFitterAndEvaluator(weather_data, 600, 200, ind_wind=True)
eval_data_all_horizons.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_600_200_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

avg_pinball_loss_qra, avg_pinball_loss_per_quantile_qra, avg_pinball_loss_overall_qra = QuantilePredictionEvaluator(eval_data_all_horizons, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_600_200_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_600_200_wind_' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

"""
QRA for weekly forecasts 
"""

preds_QRA_all = QRAQuantilePredictor(600, weather_data, ind_wind=True)
preds_QRA_all[['0.025', '0.25', '0.5', '0.75', '0.975']] = preds_QRA_all[['0.025', '0.25', '0.5', '0.75', '0.975']].apply(lambda x: np.sort(x), axis = 1, raw = True)


print(1)