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
from WeatherForecastModels import EMOS, GBM, QRF, QuantilePredictionEvaluator
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor
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
def RollingWindowQuantileCalculator(model, data, length_train_data, index_drop_na, horizon, emos_ind_boosting, considered_days):

    ind = 1

    for h in horizon:
        data_h = data[(data['fcst_hour'] == h)]
        if index_drop_na == True:
            data_h = data_h.dropna()

        data_h = data_h.reset_index()
        data_h = data_h.drop(columns='index')
        len_data = len(data_h)
        len_preds = len_data - length_train_data

        # Dataframe that contains quantile predictions for the different horizons and test data times
        quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[length_train_data:len_data-1], columns=['init_tm','obs_tm'])
        quantile_preds_rw_h['horizon'] = h
        quantile_preds_rw_h[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len_preds-1, 5))

        quantile_preds_rw_h = quantile_preds_rw_h.reset_index()
        quantile_preds_rw_h = quantile_preds_rw_h.drop(columns='index')

        for i in range(0, len_preds - 1):

            X_y_train = data_h.iloc[i:i+length_train_data]
            X_y_train = X_y_train[(X_y_train['init_tm'].apply(lambda x: x.to_pydatetime().month) <= (
                        data_h['init_tm'].iloc[i + length_train_data] + timedelta(
                        days=considered_days)).to_pydatetime().month) & (
                        X_y_train['init_tm'].apply(lambda x: x.to_pydatetime().month) >= (
                        data_h['init_tm'].iloc[i + length_train_data] + timedelta(
                        days=considered_days)).to_pydatetime().month)]
            X_train = X_y_train.drop(columns='obs')
            y_train = X_y_train['obs']
            time = data_h['init_tm'].iloc[i + length_train_data]
            X_test = data_h.drop(columns=['obs', 'init_tm', 'fcst_hour', 'obs_tm']).iloc[i + length_train_data]

            for q in quantile_levels:
                if model == EMOS:
                    quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
                        columns=['init_tm', 'obs_tm']), y_train, X_test.drop(
                        columns=['init_tm', 'obs_tm']).to_frame().T, emos_ind_boosting)

                else:
                    quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
                        columns=['init_tm', 'fcst_hour', 'obs_tm']), y_train, X_test)

        if ind == 1:
            quantile_preds_rw = quantile_preds_rw_h
        else:
            quantile_preds_rw = quantile_preds_rw.append(quantile_preds_rw_h)

        ind = ind + 1

    return quantile_preds_rw

#quantile_preds_rw_emos_submissions = RollingWindowQuantileCalculator(EMOS, weather_data, 984, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
#quantile_preds_rw_emos_submissions[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

# Calculate Quantile Predictions with Rolling Window Approach
# quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
# quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
# quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366)
# quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 900, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
#
# # Evaluate all rolling window predictions
# quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
# avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
#
# quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
# avg_pinball_loss_emos_boosting, avg_pinball_loss_per_quantile_emos_boosting, avg_pinball_loss_overall_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
# quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
# avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
# quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
# avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)

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
#     average_qs_len_months['QRF'][average_qs_len_months['nbr_considered_months'] == m] = avg_pinball_loss_overall_qrf

"""
Combination of forecasts with quantile regression and evaluation of approach with rolling window 
"""

def RollingWindowQuantileCalculatorAllModels(len_rw_models):

    quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, len_rw_models, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
    quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, len_rw_models, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)
    quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, len_rw_models, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366)
    quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, len_rw_models, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366)

    quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')

    return quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm

quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm = RollingWindowQuantileCalculatorAllModels(900)

def QRAFitterAndEvaluator(quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm, len_rw_qra_model):

    ind = 0
    for h in horizon:

        quantile_preds_rw_qrf_h = quantile_preds_rw_qrf[(quantile_preds_rw_qrf['horizon'] == h)]
        quantile_preds_rw_qrf_h = quantile_preds_rw_qrf_h.reset_index()
        quantile_preds_rw_qrf_h = quantile_preds_rw_qrf_h.drop(columns='index')

        quantile_preds_rw_emos_h = quantile_preds_rw_emos[(quantile_preds_rw_emos['horizon'] == h)]
        quantile_preds_rw_emos_h = quantile_preds_rw_emos_h.reset_index()
        quantile_preds_rw_emos_h = quantile_preds_rw_emos_h.drop(columns='index')

        quantile_preds_rw_emos_boosting_h = quantile_preds_rw_emos_boosting[(quantile_preds_rw_emos_boosting['horizon'] == h)]
        quantile_preds_rw_emos_boosting_h = quantile_preds_rw_emos_boosting_h.reset_index()
        quantile_preds_rw_emos_boosting_h = quantile_preds_rw_emos_boosting_h.drop(columns='index')

        quantile_preds_rw_gbm_h = quantile_preds_rw_gbm[(quantile_preds_rw_gbm['horizon'] == h)]
        quantile_preds_rw_gbm_h = quantile_preds_rw_gbm_h.reset_index()
        quantile_preds_rw_gbm_h = quantile_preds_rw_gbm_h.drop(columns='index')


        eval_data = quantile_preds_rw_qrf_h[['init_tm', 'obs_tm', 'obs', 'horizon']].iloc[
                    len_rw_qra_model:len(quantile_preds_rw_qrf_h)]
        eval_data[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len(eval_data), 5))

        for i in range(0,len(quantile_preds_rw_qrf_h) - len_rw_qra_model):

            for q in quantile_levels:

                X = pd.DataFrame(quantile_preds_rw_qrf_h[str(q)].iloc[i:i + len_rw_qra_model + 1])
                X = X.rename({str(q): str(q) + 'qrf' }, axis = 'columns')
                X[str(q) + 'emos'] = quantile_preds_rw_emos_h[str(q)].iloc[i:i + len_rw_qra_model + 1]
                X[str(q) + 'emos_boosting'] = quantile_preds_rw_emos_boosting_h[str(q)].iloc[i:i + len_rw_qra_model + 1]
                #X[str(q) + 'gbm'] = quantile_preds_rw_gbm_h[str(q)].iloc[i:i + len_rw_qra_model + 1]
                X_train = X.iloc[0:-1]
                X_test = X.tail(1)
                y = quantile_preds_rw_qrf_h['obs'].iloc[i:i + len_rw_qra_model + 1]
                y_train = y.iloc[0:-1]
                y_test = y.tail(1)

                qr = QuantileRegressor(quantile=q, alpha=0)
                y_pred = qr.fit(X_train, y_train).predict(X_test)
                eval_data = eval_data.reset_index()
                eval_data = eval_data.drop(columns = 'index')
                eval_data[str(q)].iloc[i] = y_pred

        if ind == 0:
            eval_data_all_horizons = eval_data
            ind = 1
        else:
            eval_data_all_horizons = eval_data_all_horizons.append(eval_data)

    #avg_pinball_loss, avg_pinball_loss_per_quantile, avg_pinball_loss_overall = QuantilePredictionEvaluator(predictions, quantile_levels, horizon)

    return eval_data_all_horizons

eval_data_all_horizons = QRAFitterAndEvaluator(quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm, 50)

eval_data_all_horizons[['0.025', '0.25', '0.5', '0.75', '0.975']] = eval_data_all_horizons[['0.025', '0.25', '0.5', '0.75', '0.975']].apply(lambda x: np.sort(x), axis = 1, raw = True)

#todo: deal with quantile crossing and find out why 15 - 17, 19 sept 2021 75 % quantile so weird



avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(eval_data_all_horizons, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)


print(1)
# kf = KFold(n_splits=8)
# # train individuals models on 4 folds, train quantile regression on 2 other folds and evaluate on 2 other folds
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

"""
Absolute Evaluation of forecasts separately to evaluate if model generally is appropriate 
"""

"""
Diebold Mariano Test for forecast performance and PIT 
"""
