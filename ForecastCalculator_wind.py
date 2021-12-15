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
from Dataloader_weather import DataUpdaterWeather, DataLoaderWeather, RealObservationsAdder
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

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
weather_data = DataUpdaterWeather('2021-12-14')

weather_data['ens_mean'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].mean(axis=1)
weather_data['ens_sd'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].std(axis=1)

df_aswdir_s, df_clct, df_mslp, df_t_2m, df_wind_10m = DataLoaderWeather(weather_data)

# add month and year as variables since temperature depends heavily on the month of the year
# and possibly a bit on the year itself due to climate change
df_wind_10m['month'] = df_wind_10m['obs_tm'].apply(lambda x: x.to_pydatetime().month)
df_wind_10m['year'] = df_wind_10m['obs_tm'].apply(lambda x: x.to_pydatetime().year)

df_wind_10m_mod = df_wind_10m[['init_tm', 'fcst_hour', 'obs_tm', 'obs', 'ens_mean', 'ens_sd', 'month']]
df_wind_10m_mod = df_wind_10m_mod.rename(columns={'ens_mean': 'ens_mean_wind_10m', 'ens_sd': 'ens_sd_wind_10m'})
df_wind_10m_mod = df_wind_10m_mod.merge(df_clct[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_wind_10m_mod = df_wind_10m_mod.rename(columns={'ens_mean': 'ens_mean_clct', 'ens_sd': 'ens_sd_clct'})
df_wind_10m_mod = df_wind_10m_mod.merge(df_mslp[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_wind_10m_mod = df_wind_10m_mod.rename(columns={'ens_mean': 'ens_mean_mslp', 'ens_sd': 'ens_sd_mslp'})
df_wind_10m_mod = df_wind_10m_mod.merge(df_t_2m[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_wind_10m_mod = df_wind_10m_mod.rename(columns={'ens_mean': 'ens_mean_t_2m', 'ens_sd': 'ens_sd_t_2m'})

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
                        train1.crch <- crch(obs ~ ens_mean|ens_sd, data = wind_10m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log",control = crch.boost(mstop = "aic"))

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
        percentile_q = norm(loc=estimated_params_boost_EMOS['mu'][estimated_params_boost_EMOS['horizon'] == i], scale=estimated_params_boost_EMOS['sd'][estimated_params_boost_EMOS['horizon'] == i]).ppf(q)
        estimated_params_boost_EMOS[str(q)][estimated_params_boost_EMOS['horizon'] == i] = percentile_q

    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

""" Quantile Gradient Boosting """

def GBM(q, X_train, y_train, X_test):
    mod = GradientBoostingRegressor(loss='quantile', alpha=q,
                                    n_estimators=15, max_depth=10,
                                    learning_rate=.01, min_samples_leaf=10,
                                    min_samples_split=15)
    # mod = GradientBoostingRegressor(loss='quantile', alpha=q,
    #                                 n_estimators=10, max_depth=5,
    #                                 learning_rate=.01, min_samples_leaf=10,
    #                                 min_samples_split=10)
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test.array.reshape(1, -1))
    return pred

estimated_quantiles = pd.DataFrame(horizon, columns=['horizon'])
estimated_quantiles[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))

for i in horizon:
    wind_data_boosting_i = df_wind_10m_mod[(df_wind_10m_mod['fcst_hour'].isin([i, i + 1, i - 1, i + 2, i - 2, i + 3, i - 3, i + 4, i - 4]))]
    wind_data_boosting_i = wind_data_boosting_i.reset_index()
    wind_data_boosting_i = wind_data_boosting_i.drop(columns=['index', 'init_tm', 'obs_tm'])
    X_y_train = wind_data_boosting_i[:-1]
    X_y_train = X_y_train.dropna()
    X_test = wind_data_boosting_i.drop(columns=['obs']).iloc[len(wind_data_boosting_i)-1]
    y_train = X_y_train['obs']
    X_train = X_y_train.drop(columns=['obs'])

    for q in quantile_levels:
        estimated_quantiles[str(q)][estimated_quantiles['horizon'] == i] = GBM(q, X_train, y_train, X_test)

print('A')
"""
Idea: systematic erros, biases etc more similar when same time of the year so only use this month and the 2 months around it for estimation 
"""