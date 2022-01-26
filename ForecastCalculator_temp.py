import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from datetime import datetime
from Dataloader_weather import DataUpdaterWeather
from WeatherForecastModels import EMOS, GBM, QRF, QuantilePredictionEvaluator, QRAFitterAndEvaluator, QRAQuantilePredictor, RollingWindowQuantileCalculator, QRAFitterAndEvaluator_multiple_alphas
from scipy.stats import norm
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
#weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather('2022-01-19', 'temperature')
weather_data, df_t_2m, df_wind_10m = DataUpdaterWeather(datetime.strftime(datetime.now(), '%Y-%m-%d'), 'temperature')
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
    #
    t2m_data_fcsth_i_train = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'ens_skewness', 'obs']].iloc[0:len(t2m_data_fcsth_i) - 1]
    t2m_data_fcsth_i_test = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'ens_skewness']].iloc[-1:]
    # weather_data_i = weather_data[(weather_data['fcst_hour'] == i)]
    # weather_data_i = weather_data_i.drop(columns = ['init_tm', 'obs_tm'])
    # t2m_data_fcsth_i_train = weather_data_i.iloc[0:len(weather_data_i) - 1]
    # t2m_data_fcsth_i_test = weather_data_i.iloc[-1:]

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
    # train1.crch <- crch(obs ~ ens_mean + ens_skewness|ens_sd + ens_skewness, data = t2m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log")
    # train1.crch <- crch(obs ~ ens_mean_t_2m + ens_skewness_t_2m + ens_mean_clct + ens_mean_mslp|ens_sd_t_2m + ens_sd_clct + ens_sd_mslp, data = t2m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log")
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
estimated_params_emos = pd.DataFrame(horizon, columns=['horizon'])
estimated_params_emos['mu'] = np.zeros(len(estimated_params))
estimated_params_emos['sd'] = np.zeros(len(estimated_params))
estimated_params_emos[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))

ind_boosting = False
for i in horizon:
    weather_data_i = weather_data[(weather_data['fcst_hour'] == i)].reset_index().drop(columns = 'index')
    weather_data_i = weather_data_i.drop(columns = ['init_tm', 'obs_tm'])
    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        estimated_params_emos[str(q)][estimated_params['horizon'] == i] = EMOS(q, weather_data_i.drop(columns='obs').iloc[0:len(weather_data_i) - 1], weather_data_i['obs'].iloc[0:len(weather_data_i) - 1], weather_data_i.drop(columns='obs').iloc[-1:], ind_boosting)

estimated_params_emos[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

estimated_params_emos_boosting = pd.DataFrame(horizon, columns=['horizon'])
estimated_params_emos_boosting['mu'] = np.zeros(len(estimated_params))
estimated_params_emos_boosting['sd'] = np.zeros(len(estimated_params))
estimated_params_emos_boosting[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))

ind_boosting = True
for i in horizon:
    weather_data_i = weather_data[(weather_data['fcst_hour'] == i)].reset_index().drop(columns = 'index')
    weather_data_i = weather_data_i.drop(columns = ['init_tm', 'obs_tm'])
    quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

    for q in quantile_levels:
        estimated_params_emos_boosting[str(q)][estimated_params['horizon'] == i] = EMOS(q, weather_data_i.drop(columns='obs').iloc[0:len(weather_data_i) - 1], weather_data_i['obs'].iloc[0:len(weather_data_i) - 1], weather_data_i.drop(columns='obs').iloc[-1:], ind_boosting)

estimated_params_emos_boosting[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)



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

quantile_preds_avg_emos_emos_boosting = pd.DataFrame(estimated_params_boost_EMOS['horizon'])
for q in quantile_levels:
    quantile_preds_avg_emos_emos_boosting[str(q)] = pd.concat([estimated_params_emos[str(q)], estimated_params_emos_boosting[str(q)]],axis=1).mean(axis = 1)

quantile_preds_avg_emos_emos_boosting[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

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
# min_sample_sizes = [10, 20]
# n_estimators = [200, 400]
# avg_pinball_loss_trees_hyperparameter = pd.DataFrame(np.zeros((len(min_sample_sizes) * len(n_estimators) * 2, 4)), columns=['model','min_sample_sizes', 'n_estimators', 'avg_quantile_loss'])
#
# i = 0
# for min_sample_size in min_sample_sizes:
#     for n_estimator in n_estimators:
#         avg_pinball_loss_trees_hyperparameter['min_sample_sizes'].iloc[i] = min_sample_size
#         avg_pinball_loss_trees_hyperparameter['min_sample_sizes'].iloc[i+1] = min_sample_size
#         avg_pinball_loss_trees_hyperparameter['n_estimators'].iloc[i] = n_estimator
#         avg_pinball_loss_trees_hyperparameter['n_estimators'].iloc[i+1] = n_estimator
#         avg_pinball_loss_trees_hyperparameter['model'].iloc[i] = 'QRF'
#         avg_pinball_loss_trees_hyperparameter['model'].iloc[i+1] = 'GBM'
#
#         quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = min_sample_size, n_estimator = n_estimator)
#         quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = min_sample_size, n_estimator = n_estimator)
#
#         quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
#                                                             on=['init_tm', 'obs_tm'], how='left', validate='1:1')
#         quantile_preds_rw_qrf = quantile_preds_rw_qrf.dropna()
#         quantile_preds_rw_qrf = quantile_preds_rw_qrf.reset_index().drop(columns = 'index')
#         avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
#         quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
#         quantile_preds_rw_gbm = quantile_preds_rw_gbm.dropna()
#         quantile_preds_rw_gbm = quantile_preds_rw_gbm.reset_index().drop(columns = 'index')
#         avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
#         avg_pinball_loss_trees_hyperparameter['avg_quantile_loss'].iloc[i] = avg_pinball_loss_overall_qrf
#         avg_pinball_loss_trees_hyperparameter['avg_quantile_loss'].iloc[i+1] = avg_pinball_loss_overall_gbm
#
#         i = i + 2

#avg_pinball_loss_trees_hyperparameter.to_csv('/Users/franziska/Dropbox/DataPTSFC/mean_quantile_scores_rw_hyperparameter_choice_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

min_sample_size_gbm = 10
min_sample_size_qrf = 10
n_estimator_gbm = 400
n_estimator_qrf = 200

#quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = 0, n_estimator = 0)
#quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = 366, min_sample_size = 0, n_estimator = 0)
#quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = min_sample_size_qrf, n_estimator = n_estimator_qrf)
#quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 750, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = min_sample_size_gbm, n_estimator = n_estimator_gbm)

# # # # Evaluate all rolling window predictions
# quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')
# quantile_preds_rw_emos = quantile_preds_rw_emos.dropna()
# quantile_preds_rw_emos = quantile_preds_rw_emos.reset_index().drop(columns = 'index')
# avg_pinball_loss_emos, avg_pinball_loss_per_quantile_emos, avg_pinball_loss_overall_emos = QuantilePredictionEvaluator(quantile_preds_rw_emos, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
#
# quantile_preds_rw_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_emos.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_emos_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# # # #
# quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how='left', validate='1:1')
# quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.dropna()
# quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.reset_index().drop(columns = 'index')
# avg_pinball_loss_emos_boosting, avg_pinball_loss_per_quantile_emos_boosting, avg_pinball_loss_overall_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
# # #
# quantile_preds_rw_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_emos_boosting_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_emos_boosting_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# # # avg_pinball_loss_overall_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_emos_boosting_700' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# # #
# quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
# quantile_preds_rw_gbm = quantile_preds_rw_gbm.dropna()
# quantile_preds_rw_gbm = quantile_preds_rw_gbm.reset_index().drop(columns = 'index')
# avg_pinball_loss_gbm, avg_pinball_loss_per_quantile_gbm, avg_pinball_loss_overall_gbm = QuantilePredictionEvaluator(quantile_preds_rw_gbm, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
# # # #
# quantile_preds_rw_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_gbm_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_gbm_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_gbm.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_gbm_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

# quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']], on=['init_tm', 'obs_tm'], how='left', validate='1:1')
# quantile_preds_rw_qrf = quantile_preds_rw_qrf.dropna()
# quantile_preds_rw_qrf = quantile_preds_rw_qrf.reset_index().drop(columns = 'index')
# avg_pinball_loss_qrf, avg_pinball_loss_per_quantile_qrf, avg_pinball_loss_overall_qrf = QuantilePredictionEvaluator(quantile_preds_rw_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
#
# quantile_preds_rw_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qrf_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qrf_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_overall_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_overall_qrf_750' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

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
#     quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m, min_sample_size = 0, n_estimator = 0)
#     quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=True, considered_days = m, min_sample_size = 0, n_estimator = 0)
#     quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, 600, index_drop_na=True, horizon=horizon, emos_ind_boosting=False, considered_days = m, min_sample_size = min_sample_size_gbm, n_estimator = n_estimator_gbm)
#     quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, 600, index_drop_na=True, horizon=horizon,
#                                                             emos_ind_boosting=False, considered_days=m, min_sample_size = min_sample_size_qrf, n_estimator = n_estimator_qrf)
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
#average_qs_len_months.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_average_qs_len_months_600_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

"""
Forecast Combination by (simple) averaging 
"""
# read rolling window predictions in
# quantile_preds_rw_emos = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_7502022-01-11.csv')
# quantile_preds_rw_emos_boosting = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_emos_boosting_7502022-01-11.csv')
# quantile_preds_rw_gbm = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_gbm_7502022-01-25.csv')
# quantile_preds_rw_qrf = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_qrf_7502022-01-25.csv')
#
# # ALL MODELS
# quantile_preds_rw_avg_all_models = pd.DataFrame(quantile_preds_rw_emos[['init_tm', 'obs_tm', 'horizon', 'obs']])
# for q in quantile_levels:
#     quantile_preds_rw_avg_all_models[str(q)] = pd.concat([quantile_preds_rw_emos[str(q)],quantile_preds_rw_emos_boosting[str(q)], quantile_preds_rw_gbm[str(q)], quantile_preds_rw_qrf[str(q)]],axis=1).mean(axis = 1)
#
# avg_pinball_loss_avg_all_models, avg_pinball_loss_per_quantile_avg_all_models, avg_pinball_loss_overall_avg_all_models = QuantilePredictionEvaluator(quantile_preds_rw_avg_all_models, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
# #
# quantile_preds_rw_avg_all_models.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_avg_all_models_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_avg_all_models.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_avg_all_models_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_avg_all_models.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_avg_all_models_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
# # TWO BEST MODELS; ONE PARAMETRIC ONE NON-PARAMETRIC
# quantile_preds_rw_avg_emos_qrf = pd.DataFrame(quantile_preds_rw_emos[['init_tm', 'obs_tm', 'horizon', 'obs']])
# for q in quantile_levels:
#     quantile_preds_rw_avg_emos_qrf[str(q)] = pd.concat([quantile_preds_rw_emos[str(q)], quantile_preds_rw_qrf[str(q)]],axis=1).mean(axis = 1)
#
# avg_pinball_loss_avg_emos_qrf, avg_pinball_loss_per_quantile_avg_emos_qrf, avg_pinball_loss_overall_avg_emos_qrf = QuantilePredictionEvaluator(quantile_preds_rw_avg_emos_qrf, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
# #
# quantile_preds_rw_avg_emos_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_avg_emos_qrf_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_avg_emos_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_avg_emos_qrf_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_avg_emos_qrf.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_avg_emos_qrf_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
#
# # TWO BEST MODELS OVERALL
# quantile_preds_rw_avg_emos_emos_boosting = pd.DataFrame(quantile_preds_rw_emos[['init_tm', 'obs_tm', 'horizon', 'obs']])
# for q in quantile_levels:
#     quantile_preds_rw_avg_emos_emos_boosting[str(q)] = pd.concat([quantile_preds_rw_emos[str(q)], quantile_preds_rw_emos_boosting[str(q)]],axis=1).mean(axis = 1)
#
# avg_pinball_loss_avg_emos_emos_boosting, avg_pinball_loss_per_quantile_avg_emos_emos_boosting, avg_pinball_loss_overall_avg_emos_emos_boosting = QuantilePredictionEvaluator(quantile_preds_rw_avg_emos_emos_boosting, quantile_levels=[0.025, 0.25, 0.5, 0.75, 0.975], horizons=horizon)
# #
# quantile_preds_rw_avg_emos_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/quantile_preds_rw_avg_emos_emos_boosting_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_avg_emos_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_avg_emos_emos_boosting_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)
# avg_pinball_loss_per_quantile_avg_emos_emos_boosting.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_avg_emos_emos_boosting_750_' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

"""
Combination of forecasts with quantile regression and evaluation of approach with rolling window 
"""
alphas = [0.05, 0.1, 0.15, 0.2]
avg_pinball_loss_alphas = pd.DataFrame(alphas, columns=['alpha'])

eval_data_all_horizons_005, eval_data_all_horizons_01, eval_data_all_horizons_015, eval_data_all_horizons_02  = QRAFitterAndEvaluator_multiple_alphas(weather_data, 700, 50, ind_wind=False, alphas = alphas)

eval_data_all_horizons_005.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_700_50_alpha_005_'+ datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

avg_pinball_loss_qra_005, avg_pinball_loss_per_quantile_qra_005, avg_pinball_loss_overall_qra_005 = QuantilePredictionEvaluator(eval_data_all_horizons_005, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra_005.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_700_50_005' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra_005.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50_005' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

eval_data_all_horizons_01.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_700_50_alpha_01_'+ datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

avg_pinball_loss_qra_01, avg_pinball_loss_per_quantile_qra_01, avg_pinball_loss_overall_qra_01 = QuantilePredictionEvaluator(eval_data_all_horizons_01, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra_01.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_700_50_01' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra_01.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50_01' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

eval_data_all_horizons_015.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_700_50_alpha_015_'+ datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

avg_pinball_loss_qra_015, avg_pinball_loss_per_quantile_qra_015, avg_pinball_loss_overall_qra_015 = QuantilePredictionEvaluator(eval_data_all_horizons_015, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra_015.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_700_50_015' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra_015.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50_015' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

eval_data_all_horizons_02.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_700_50_alpha_02_'+ datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

avg_pinball_loss_qra_02, avg_pinball_loss_per_quantile_qra_02, avg_pinball_loss_overall_qra_02 = QuantilePredictionEvaluator(eval_data_all_horizons_02, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
avg_pinball_loss_qra_02.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_700_50_02' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
avg_pinball_loss_per_quantile_qra_02.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50_02' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)


avg_pinball_loss_alphas[avg_pinball_loss_alphas['alpha'] == 0.05]['avg_pinball_loss'] = avg_pinball_loss_overall_qra_005
avg_pinball_loss_alphas[avg_pinball_loss_alphas['alpha'] == 0.01]['avg_pinball_loss'] = avg_pinball_loss_overall_qra_01
avg_pinball_loss_alphas[avg_pinball_loss_alphas['alpha'] == 0.15]['avg_pinball_loss'] = avg_pinball_loss_overall_qra_015
avg_pinball_loss_alphas[avg_pinball_loss_alphas['alpha'] == 0.2]['avg_pinball_loss'] = avg_pinball_loss_overall_qra_02

avg_pinball_loss_alphas.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50_all_alphas_' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)


# eval_data_all_horizons = QRAFitterAndEvaluator(weather_data, 700, 50, ind_wind=False)
# eval_data_all_horizons.to_csv('/Users/franziska/Dropbox/DataPTSFC/eval_data_all_horizons_qra_rw_700_50' + datetime.strftime(datetime.now(), '%Y-%m-%d') + '.csv', index=False)

# avg_pinball_loss_qra, avg_pinball_loss_per_quantile_qra, avg_pinball_loss_overall_qra = QuantilePredictionEvaluator(eval_data_all_horizons, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)
# avg_pinball_loss_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_qra_rw_700_50' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
# avg_pinball_loss_per_quantile_qra.to_csv('/Users/franziska/Dropbox/DataPTSFC/avg_pinball_loss_per_quantile_qra_rw_700_50' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

"""
QRA for weekly forecasts 
"""

preds_QRA_all = QRAQuantilePredictor(600, weather_data, ind_wind = False)

print(1)

"""
Absolute Evaluation of forecasts separately to evaluate if model generally is appropriate 
"""

"""
Diebold Mariano Test for forecast performance and PIT 
"""
#  Check with Diebold Mariano Test if performance of QRA forecasts significantly outperforms the performance of the best individual models forecasts
