import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from datetime import datetime,timedelta
from Dataloader_weather import DataUpdaterWeather, DataLoaderWeather
from skgarden import RandomForestQuantileRegressor
from scipy.stats import norm, kurtosis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

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
#weather_data = DataUpdaterWeather(datetime.strftime(datetime.now(), '%Y-%m-%d'))
weather_data = DataUpdaterWeather('2021-12-08')

# for each variable add the ensemble mean and standard deviation
weather_data['ens_mean'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].mean(axis=1)
weather_data['ens_sd'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].std(axis=1)
#weather_data['ens_kurtosis'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].apply(lambda x: kurtosis(x))
# split large weather_data DataFrame in smaller DataFrames for the different weather variables
df_aswdir_s, df_clct, df_mslp, df_t_2m, df_wind_10m = DataLoaderWeather(weather_data)

# add month and year as variables since temperature depends heavily on the month of the year
# and possibly a bit on the year itself due to climate change
df_t_2m['month'] = df_t_2m['obs_tm'].apply(lambda x: x.to_pydatetime().month)
df_t_2m['year'] = df_t_2m['obs_tm'].apply(lambda x: x.to_pydatetime().year)

df_t_2m_mod = df_t_2m[['init_tm', 'fcst_hour', 'obs_tm', 'obs', 'ens_mean', 'ens_sd', 'month']]
df_t_2m_mod = df_t_2m_mod.rename(columns={'ens_mean': 'ens_mean_t_2m', 'ens_sd': 'ens_sd_t_2m'})
df_t_2m_mod = df_t_2m_mod.merge(df_clct[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_t_2m_mod = df_t_2m_mod.rename(columns={'ens_mean': 'ens_mean_clct', 'ens_sd': 'ens_sd_clct'})
df_t_2m_mod = df_t_2m_mod.merge(df_mslp[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_t_2m_mod = df_t_2m_mod.rename(columns={'ens_mean': 'ens_mean_mslp', 'ens_sd': 'ens_sd_mslp'})
df_t_2m_mod = df_t_2m_mod.merge(df_wind_10m[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd']],
                                              how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
df_t_2m_mod = df_t_2m_mod.rename(columns={'ens_mean': 'ens_mean_wind_10m', 'ens_sd': 'ens_sd_wind_10m'})

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

"""
Reprogram R example
"""

horizon = [36, 48, 60, 72, 84]

def EMOS_QuantileEstimatorRollingWindow(horizon, data, len_train_data):

    data_len = data[(data['fcst_hour'] == horizon[0])]
    estimated_params = pd.DataFrame(np.nan * np.zeros((len(data_len) - len_train_data + 10, 10)),
                                    columns=["mu_" + str(i) for i in horizon] + ["sd_" + str(i) for i in horizon])
    for i in horizon:
        start_time_train = data['init_tm'].iloc[0]
        end_time_train = start_time_train + timedelta(days=len_train_data)
        data_i = data[(data['fcst_hour'] == i)]
        for j in range(0, len(data) - len_train_data):
            # if j >= 12:
            #     print('stop')
            data_train = data_i[(start_time_train <= data_i['init_tm']) & (data_i['init_tm'] <= end_time_train)].reset_index()
            data_test = data_i[data_i['init_tm'] == end_time_train + timedelta(days=1)].reset_index()
            if len(data_test) == 1:
                data_train_i = data_train[['init_tm','obs_tm', 'ens_mean', 'ens_sd', 'obs']]
                data_test_i = data_test[['init_tm', 'obs_tm', 'ens_mean', 'ens_sd', 'obs']]

                with localconverter(robjects.default_converter + pandas2ri.converter):
                    data_train_i_r = robjects.conversion.py2rpy(data_train_i)
                    data_test_i_r = robjects.conversion.py2rpy(data_test_i)

                pandas2ri.activate()
                robjects.globalenv['data_train_i_r'] = data_train_i
                robjects.r('''
                           f <- function(data_train_i) {
            
                                    library(crch)
                                    train1.crch <- crch(obs ~ ens_mean|ens_sd, data = data_train_i_r, dist = "gaussian", type = "crps", link.scale = "log")
            
                            }
                            ''')
                r_f = robjects.globalenv['f']
                rf_model = (r_f(data_train_i_r))
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
                prediction_mu = (r_g(rf_model, data_test_i_r)).values
                prediction_sd = (r_h(rf_model, data_test_i_r)).values

                estimated_params['mu_' + str(i)].iloc[j] = prediction_mu
                estimated_params['sd_' + str(i)].iloc[j] = prediction_sd
            else:
                estimated_params['mu_' + str(i)].iloc[j] = np.NaN
                estimated_params['sd_' + str(i)].iloc[j] = np.NaN
        #    estimated_params['crps'][estimated_params['horizon'] == i] = score

            # quantile_levels = [0.025,0.25,0.5,0.75,0.975]
            #
            # for q in quantile_levels:
            #     percentile_q = norm(loc=estimated_params['mu'][estimated_params['horizon'] == i], scale=estimated_params['sd'][estimated_params['horizon'] == i]).ppf(q)
            #     estimated_params[str(q)][estimated_params['horizon'] == i] = percentile_q
            #
            start_time_train = start_time_train + timedelta(days=1)
            end_time_train = end_time_train + timedelta(days=1)
            print(j)
        print(i)

    return estimated_params

#estimated_params_emos_rw = EMOS_QuantileEstimatorRollingWindow(horizon, df_t_2m, 365)

print(1)
    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
#estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

# with localconverter(robjects.default_converter + pandas2ri.converter):
#  t2m_data_fcsth48_obs_r = robjects.conversion.py2rpy(t2m_data_fcsth48['obs'])
#  t2m_data_fcsth48_obs_r_2 = robjects.vectors.FloatVector(t2m_data_fcsth48['obs'])
#  t2m_data_fcsth48_ens_r = robjects.conversion.py2rpy(t2m_data_fcsth48[["ens_" + str(i) for i in range(1, 41)]])
#  t2m_data_fcsth48_ens_r_2 = robjects.vectors.FloatVector(t2m_data_fcsth48[["ens_" + str(i) for i in range(1, 41)]].values)

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
    #t2m_data_fcsth_i = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'].dt.month.isin([10,11,12])]

    t2m_data_fcsth_i_train = t2m_data_fcsth_i[['ens_mean', 'ens_sd', 'obs']].iloc[0:len(t2m_data_fcsth_i) - 1]
    t2m_data_fcsth_i_test = t2m_data_fcsth_i[['ens_mean', 'ens_sd']].iloc[-1:]

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
                        train1.crch <- crch(obs ~ ens_mean|ens_sd, data = t2m_data_fcsth_i_train_r, dist = "gaussian", type = "crps", link.scale = "log")

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

""" EMOS and boosting """
estimated_params_boost_EMOS = pd.DataFrame(horizon, columns=['horizon'])
estimated_params_boost_EMOS['mu'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS['sd'] = np.zeros(len(estimated_params_boost_EMOS))
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params_boost_EMOS))

for i in horizon:
    t2m_data_fcsth_i = df_t_2m[(df_t_2m['fcst_hour'] == i)]
    t2m_data_fcsth_i = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'].apply(lambda x: x.to_pydatetime().month).isin([11,12,1])]
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
    # crch.boost(maxit=100, nu=0.1, start=NULL, dot="separate",
    #            mstop=c("max", "aic", "bic", "cv"), nfolds=10, foldid=NULL,
    #            maxvar=NULL)
    # train1.crch < - crch.boost.fit(obs
    # ~ ens_mean | ens_sd, dist = "gaussian", type = "crps", link.scale = "log")

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

    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)
estimated_params_boost_EMOS[['0.025', '0.25', '0.5', '0.75', '0.975']].to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)

""" Quantile Gradient Boosting """

def GBM(q, X_train, y_train, X_test):
    mod = GradientBoostingRegressor(loss='quantile', alpha=q,
                                    n_estimators=15, max_depth=8,
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
    temp_data_boosting_i = df_t_2m_mod[(df_t_2m_mod['fcst_hour'].isin([i, i + 1, i - 1, i + 2, i - 2, i + 3, i - 3, i + 4, i - 4]))]
    temp_data_boosting_i = temp_data_boosting_i.reset_index()
    temp_data_boosting_i = temp_data_boosting_i.drop(columns=['index', 'init_tm', 'obs_tm'])
    X_y_train = temp_data_boosting_i[:-1]
    X_y_train = X_y_train.dropna()
    X_test = temp_data_boosting_i.drop(columns=['obs']).iloc[len(temp_data_boosting_i)-1]
    y_train = X_y_train['obs']
    X_train = X_y_train.drop(columns=['obs'])

    for q in quantile_levels:
        estimated_quantiles[str(q)][estimated_quantiles['horizon'] == i] = GBM(q, X_train, y_train, X_test)

"""
Functions for rolling window approach 
"""

def RollingWindowQuantileCalculator(data, length_train_data, index_drop_na, horizon_i):

    data = data[(data['fcst_hour'] == horizon_i)]
    if index_drop_na == True:
        data = data.dropna()

    data = data.reset_index()
    data = data.drop(columns='index')
    len_data = len(data)
    len_preds = len_data - length_train_data

    # Dataframe that contains quantile predictions for the different horizons and test data times
    quantile_preds_rw = pd.DataFrame(data[['init_tm', 'obs_tm']].iloc[length_train_data:len_data-1] , columns=['init_tm','obs_tm'])
    quantile_preds_rw['horizon'] = horizon_i
    quantile_preds_rw[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len_preds-1, 5))

    quantile_preds_rw = quantile_preds_rw.reset_index()
    quantile_preds_rw = quantile_preds_rw.drop(columns='index')

    for i in range(0, len_preds - 1):
        X_y_train = data.iloc[i:i+length_train_data]
        X_train = X_y_train.drop(columns='obs')
        y_train = X_y_train['obs']
        time = data['init_tm'].iloc[i + length_train_data]
        X_test = data.drop(columns=['obs', 'init_tm', 'fcst_hour', 'obs_tm']).iloc[i + length_train_data]

        for q in quantile_levels:
            quantile_preds_rw[str(q)][quantile_preds_rw['init_tm'] == time] = GBM(q, X_train.drop(columns=['init_tm', 'fcst_hour', 'obs_tm']), y_train, X_test)

    return quantile_preds_rw

index_drop_na = True
# quantile_preds_rw = RollingWindowQuantileCalculator(df_t_2m_mod, 365, index_drop_na, 36)
#quantile_preds_rw = RollingWindowQuantileCalculator(df_t_2m_mod, 800, index_drop_na, 36)

ind = 1
for h in horizon:
    if ind == 1:
        quantile_preds_rw = RollingWindowQuantileCalculator(df_t_2m_mod, 820, index_drop_na, h)
    else:
        quantile_preds_rw_i = RollingWindowQuantileCalculator(df_t_2m_mod, 820, index_drop_na, h)
        quantile_preds_rw = quantile_preds_rw.append(quantile_preds_rw_i)

    ind = ind + 1

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

quantile_preds_rw = quantile_preds_rw.merge(df_t_2m_mod[['obs', 'init_tm', 'obs_tm']], on = ['init_tm', 'obs_tm'], how = 'left', validate = '1:1')

avg_pinball_loss, avg_pinball_loss_per_quantile, avg_pinball_loss_overall = QuantilePredictionEvaluator(quantile_preds_rw, quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975], horizons = horizon)

"""
Grid search of parameters with rolling window evaluation 
"""

rfqr = RandomForestQuantileRegressor(
    random_state=0, min_samples_split=10, n_estimators=1000)
X_train = pd.concat([df_t_2m['ens_mean'][:-1], df_t_2m['obs'][:-1]], axis=1)
X_train = X_train.dropna()
rfqr_fit = rfqr.fit(X_train['ens_mean'].array.reshape(-1,1) , X_train['obs'].values.ravel())
print('a')
rfqr_pred = rfqr.predict(df_t_2m['ens_mean'][len(df_t_2m)-1], quantile=98.5)