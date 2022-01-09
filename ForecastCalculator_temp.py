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
from WeatherForecastModels import EMOS, GBM, QRF, QuantilePredictionEvaluator, QRAFitterAndEvaluator, QRAQuantilePredictor, RollingWindowQuantileCalculator
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

"""
In the following we first set up the rpy2 framework in order to be able to use R packages afterwards
"""
base = importr('base')
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

"""
Import of R packages which are used later on with rpy2
"""
# R package names which are required in the following and therefore now installed first
packnames = ('ggplot2', 'hexbin', 'scoringRules', 'rdwd')
from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

xfun = importr('xfun')
scoringRules = rpackages.importr('scoringRules')
crch = rpackages.importr('crch')

""" 
load weather data 
"""
# load data frame weather_data with r_icon_eps weather data and ensemble forecasts from 'YYYY-MM-DD'
weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather('2021-12-14', 'temperature')
#weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather(datetime.strftime(datetime.now(), '%Y-%m-%d'), 'temperature')
"""
First visualize real temperature observations to get a feeling for the data
"""
logging.info('Starting visualization of temperature data')

df_t_2m_plot = df_t_2m.dropna()
ind = 1
for year in df_t_2m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year).unique():
    plt.plot(df_t_2m_plot['obs_tm'][df_t_2m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year) == year],
             df_t_2m_plot['obs'][df_t_2m_plot['obs_tm'].apply(lambda x: x.to_pydatetime().year) == year])
    plt.xlabel('time')
    plt.ylabel('temperature in degree celcius')
    ind = ind + 1
    plt.show()
    plt.savefig('/Users/franziska/Dropbox/DataPTSFC/Plots/' + str(year) + 'timeseries_raw_data.png')


# with localconverter(robjects.default_converter + pandas2ri.converter):
#    scoringRules.crps_sample(y = t2m_data_fcsth48_obs_r, dat = t2m_data_fcsth48[["ens_" + str(i) for i in range(1, 41)]])

# plot histogram of ensemble forecasts

"""
Reprogram R example
"""

horizon = [36, 48, 60, 72, 84]

estimated_params = pd.DataFrame(horizon, columns=['horizon'])
estimated_params['mu'] = np.zeros(len(estimated_params))
estimated_params['sd'] = np.zeros(len(estimated_params))
estimated_params['crps'] = np.zeros(len(estimated_params))
estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))

for i in horizon:
    t2m_data_fcsth_i = df_t_2m[(df_t_2m['fcst_hour'] == i)]

    t2m_data_fcsth_i_train = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'ens_skewness', 'obs']].iloc[0:len(t2m_data_fcsth_i) - 1]
    t2m_data_fcsth_i_test = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'ens_skewness']].iloc[-1:]

    # t2m_data_fcsth48.rolling(7)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        t2m_data_fcsth_i_train_r = robjects.conversion.py2rpy(t2m_data_fcsth_i_train)
        t2m_data_fcsth_i_test_r = robjects.conversion.py2rpy(t2m_data_fcsth_i_test)

    pandas2ri.activate()
    robjects.globalenv['t2m_data_fcsth_i_train_r'] = t2m_data_fcsth_i_train
    robjects.r('''
               f <- function(t2m_data_fcsth_i_train) {

                        library(crch)
                        train1.crch <- crch(obs ~ ens_mean + ens_skewness|ens_sd + ens_skewness, data = t2m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log")

                }
                ''')
    r_f = robjects.globalenv['f']
    rf_model = (r_f(t2m_data_fcsth_i_train_r))
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
    prediction_mu = (r_g(rf_model, t2m_data_fcsth_i_test_r)).values
    prediction_sd = (r_h(rf_model, t2m_data_fcsth_i_test_r)).values
#    crps_fun = scoringRules.crps
#    r_float = robjects.vectors.FloatVector
#    y_true_r = r_float(t2m_data_fcsth_i_test['obs'])
#    mu_r = r_float(prediction_mu)
#    sigma_r = r_float(prediction_sd)
#    score = scoringRules.crps(y_true_r, mean=mu_r, sd=sigma_r, family="normal")
    #    mean_crps_score = np.array(score).mean()

    estimated_params['mu'][estimated_params['horizon'] == i] = prediction_mu
    estimated_params['sd'][estimated_params['horizon'] == i] = prediction_sd
#    estimated_params['crps'][estimated_params['horizon'] == i] = score

    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        percentile_q = norm(loc=estimated_params['mu'][estimated_params['horizon'] == i], scale=estimated_params['sd'][estimated_params['horizon'] == i]).ppf(q)
        estimated_params[str(q)][estimated_params['horizon'] == i] = percentile_q

    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
print('b')
ind_boosting = False
for i in horizon:
    weather_data_i = weather_data[(weather_data['fcst_hour'] == i)].reset_index().drop(columns = 'index')
    weather_data_i = weather_data_i.drop(columns = ['init_tm', 'obs_tm'])
    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        estimated_params[str(q)][estimated_params['horizon'] == i] = EMOS(q, weather_data_i.drop(columns='obs').iloc[0:len(weather_data_i) - 1], weather_data_i['obs'].iloc[0:len(weather_data_i) - 1], weather_data_i.drop(columns='obs').iloc[-1:], ind_boosting)

estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
print('b')

""" EMOS and boosting """
estimated_params_boost_EMOS = pd.DataFrame(horizon, columns=['horizon'])
estimated_params_boost_EMOS['mu'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS['sd'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params_boost_EMOS))

for i in horizon:
    t2m_data_fcsth_i = df_t_2m[(df_t_2m['fcst_hour'] == i)]
    t2m_data_fcsth_i = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'].apply(lambda x: x.to_pydatetime().month).isin([11, 12, 1])]
    #t2m_data_fcsth_i = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'].dt.month.isin([11,12,1])]

    t2m_data_fcsth_i_train = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'obs']].iloc[0:len(t2m_data_fcsth_i) - 1]
    t2m_data_fcsth_i_test = t2m_data_fcsth_i[['ens_mean', 'ens_sd']].iloc[-1:]

    #t2m_data_fcsth_i_train = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'obs']].iloc[0:len(t2m_data_fcsth_i) - 1]
    #t2m_data_fcsth_i_test = t2m_data_fcsth_i[['ens_mean', 'ens_sd']].iloc[-1:]

    # t2m_data_fcsth48[(t2m_data_fcsth48.isnull().values) == True]
    # t2m_data_fcsth48.rolling(7)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        t2m_data_fcsth_i_train_r = robjects.conversion.py2rpy(t2m_data_fcsth_i_train)
        t2m_data_fcsth_i_test_r = robjects.conversion.py2rpy(t2m_data_fcsth_i_test)

    pandas2ri.activate()
    robjects.globalenv['t2m_data_fcsth_i_train_r'] = t2m_data_fcsth_i_train
    robjects.r('''
               f <- function(t2m_data_fcsth_i_train) {

                        library(crch)
                        train1.crch <- crch(obs ~ ens_mean|ens_sd, data = t2m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log",control = crch.boost(mstop = "aic"))

                }
                ''')

    r_f = robjects.globalenv['f']
    rf_model = (r_f(t2m_data_fcsth_i_train_r))
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
    prediction_mu = (r_g(rf_model, t2m_data_fcsth_i_test_r)).values
    prediction_sd = (r_h(rf_model, t2m_data_fcsth_i_test_r)).values
#    crps_fun = scoringRules.crps
#    r_float = robjects.vectors.FloatVector
#    y_true_r = r_float(t2m_data_fcsth_i_test['obs'])
#    mu_r = r_float(prediction_mu)
#    sigma_r = r_float(prediction_sd)
#    score = scoringRules.crps(y_true_r, mean=mu_r, sd=sigma_r, family="normal")
    #    mean_crps_score = np.array(score).mean()

    estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i] = prediction_mu
    estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i] = prediction_sd
#    estimated_params['crps'][estimated_params['horizon'] == i] = score

    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        percentile_q = norm(loc=estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i], scale=estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i]).ppf(q)
        estimated_params_boost_EMOS[str(q)][estimated_params_boost_EMOS['horizon'] == i] = percentile_q

estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

"""
Variable selection for EMOS: Want to use more regressors but must be carefull due to overfitting so only incorporate relevant ones
Therefore for now we simply look at correlations between temperature observation, temperature mean and temperature standard deviation and all other features to get feeling for dependencies 
Note: Of course using all data is kind of overfitting when we evaluate models later on with rolling window but since this is only an indicator we accept that for now
"""
corr_weather_data = weather_data.drop(columns = ['init_tm', 'obs_tm', 'month', 'year']).corr()

"""
Relative evaluation of the different models with each other to check which model performs best in terms of evaluation criterion
Use average over linear quantile scores since that's an approximation of the CRPS which is a strictly proper scoring rule
"""

#quantile_preds_rw_emos_submissions = RollingWindowQuantileCalculator(EMOS, weather_data, 984, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
#quantile_preds_rw_emos_submissions[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

# # Calculate Quantile Predictions with Rolling Window Approach # 600
quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366)
quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
# #
# # # Evaluate all rolling window predictions
quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
quantile_preds_rw_emos = quantile_preds_rw_emos.dropna()
quantile_preds_rw_emos = quantile_preds_rw_emos.reset_index().drop(columns = 'index')
avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
#
quantile_preds_rw_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_emos_700' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.dropna()
quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.reset_index().drop(columns = 'index')
avg_pinball_loss_emos_boosting, avg_pinball_loss_per_quantile_emos_boosting, avg_pinball_loss_overall_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
quantile_preds_rw_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_boosting_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_boosting_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_emos_boosting_700' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
quantile_preds_rw_gbm = quantile_preds_rw_gbm.dropna()
quantile_preds_rw_gbm = quantile_preds_rw_gbm.reset_index().drop(columns = 'index')
avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
quantile_preds_rw_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_gbm_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_gbm_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_gbm_700' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
quantile_preds_rw_qrf = quantile_preds_rw_qrf.dropna()
quantile_preds_rw_qrf = quantile_preds_rw_qrf.reset_index().drop(columns = 'index')
avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
quantile_preds_rw_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qrf_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
avg_pinball_loss_per_quantile_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qrf_600' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#avg_pinball_loss_overall_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_qrf_700' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

"""
Grid search of parameters with rolling window evaluation 
"""
# TEST IF FORECAST PERFORMANCE IMPROVES IF ONLY DATA FROM MONTHS SIMILAR TO MONTH FOR WHICH CURRENT QUANTILES ARE
# PREDICTED IS USED FOR MODEL TRAINING BECAUSE BIASES AND VARIABILITY COULD DEPEND ON TIME OF YEAR
# Note: of course this causes different training data sizes based on date that's predicted but since it's consistently
# done for all models we can compare them
# Alternative would be shorter training data sizes because then we only choose last few days/months but we wouldn't choose from months that come after month of current prediction

# array_len_days = [92, 155, 220, 366]
# average_qs_len_months = pd.DataFrame(array_len_days, columns=['nbr_considered_days'])
# average_qs_len_months[['EMOS', 'EMOS_boosting', 'Boosting', 'QRF']] = np.zeros((len(average_qs_len_months), 4))
#
# for m in array_len_days:
#
#     quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m)
#     quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = m)
#     quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m)
#     quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 600, index_drop_na=True, horizon=horizon,
#                                                             emos_ind_boosting=False, considered_days=m)
#
#     quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
#     avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
#     average_qs_len_months['EMOS'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_emos
#
#     quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
#     avg_pinball_loss_emos_boosting, avg_pinball_loss_per_quantile_emos_boosting, avg_pinball_loss_overall_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#     average_qs_len_months['EMOS_boosting'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_emos_boosting
#
#     quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
#     avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#     average_qs_len_months['Boosting'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_gbm
#
#     quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
#     avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#average_qs_len_months['QRF'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_qrf

"""
Combination of forecasts with quantile regression and evaluation of approach with rolling window 
"""

eval_data_all_horizons = QRAFitterAndEvaluator(600, 200)
eval_data_all_horizons.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_600_200' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
# #todo: deal with quantile crossing and find out why 15 - 17, 19 sept 2021 75 % quantile so weird
#
avg_pinball_loss_qra, avg_pinball_loss_per_quantile_qra, avg_pinball_loss_overall_qra = QuantilePredictionEvaluator(eval_data_all_horizons, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_600_200' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_600_200' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

"""
QRA for weekly forecasts 
"""

preds_QRA_all = QRAQuantilePredictor(600, weather_data)
preds_QRA_all[['0.025', '0.25', '0.5', '0.75', '0.975']] = preds_QRA_all[['0.025', '0.25', '0.5', '0.75', '0.975']].apply(lambda x: np.sort(x), axis = 1, raw = True)

print(1)

"""
Absolute Evaluation of forecasts separately to evaluate if model generally is appropriate 
"""

"""
Diebold Mariano Test for forecast performance and PIT 
"""
#  Check with Diebold Mariano Test if performance of QRA forecasts significantly outperforms the performance of the best individual models forecasts
