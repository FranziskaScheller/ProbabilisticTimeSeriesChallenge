import numpy
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from rpy2.robjects import pandas2ri
from scipy.stats import norm
from rpy2.robjects.conversion import localconverter
from datetime import datetime,timedelta
from Dataloader_weather import DataUpdaterWeather, DataLoaderWeather
from skgarden import RandomForestQuantileRegressor
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
""" load weather data """
full_weather_data = DataUpdaterWeather('2021-12-08')

df_aswdir_s, df_clct, df_mslp, df_t_2m, df_wind_10m = DataLoaderWeather(full_weather_data)
#df_t_2m = df_t_2m.dropna()
#df_wind_10m = df_wind_10m.dropna()
"""
First visualize real temperature observations to get a feeling for the data
"""
logging.info('Starting visualization of temperature data')

ind = 1
for year in df_t_2m['obs_tm_h'].dt.year.unique():
    plt.plot(df_t_2m['obs_tm_h'][df_t_2m['obs_tm_h'].dt.year == year],
             df_t_2m['obs'][df_t_2m['obs_tm_h'].dt.year == year])
    plt.xlabel('time')
    plt.ylabel('temperature in degree celcius')
    ind = ind + 1
    plt.show()
    plt.savefig('/Users/franziska/Dropbox/DataPTSFC/Plots/' + str(year) + 'timeseries_raw_data.png')

df_t_2m['ens_mean'] = df_t_2m[["ens_" + str(i) for i in range(1, 41)]].mean(axis=1)
df_t_2m['ens_var'] = df_t_2m[["ens_" + str(i) for i in range(1, 41)]].var(axis=1)
df_t_2m['ens_sd'] = np.sqrt(df_t_2m['ens_var'])
df_t_2m['init_tm'] = df_t_2m['init_tm'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
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

    #t2m_data_fcsth_i = t2m_data_fcsth_i.dropna()
    # t2m_data_fcsth48 = t2m_data_fcsth48.set_index(t2m_data_fcsth48['init_tm'])
    # t2m_data_fcsth_i_train = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'] <= '2020-10-24']
    # t2m_data_fcsth_i_test = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'] > '2020-10-24']

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
    t2m_data_fcsth_i = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'].dt.month.isin([11,12,1])]

    #t2m_data_fcsth_i = t2m_data_fcsth_i.dropna()
    # t2m_data_fcsth48 = t2m_data_fcsth48.set_index(t2m_data_fcsth48['init_tm'])
    # t2m_data_fcsth_i_train = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'] <= '2020-10-24']
    # t2m_data_fcsth_i_test = t2m_data_fcsth_i[t2m_data_fcsth_i['init_tm'] > '2020-10-24']

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

""" quantile random forests """

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor
#
# boston = load_boston()
# X, y = boston.data, boston.target
# kf = KFold(n_splits=5)
# rfqr = RandomForestQuantileRegressor(
#     random_state=0, min_samples_split=10, n_estimators=1000)
#
# y_true_all = []
# lower = []
# upper = []
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test, y_train, y_test = (
#         X[train_index], X[test_index], y[train_index], y[test_index])
#
#     rfqr.set_params(max_features=X_train.shape[1] // 3)
#     rfqr.fit(X_train, y_train)
#     y_true_all = np.concatenate((y_true_all, y_test))
#     upper = np.concatenate((upper, rfqr.predict(X_test, quantile=98.5)))
#     lower = np.concatenate((lower, rfqr.predict(X_test, quantile=2.5)))
#
# interval = upper - lower
# sort_ind = np.argsort(interval)
# y_true_all = y_true_all[sort_ind]
# upper = upper[sort_ind]
# lower = lower[sort_ind]
# mean = (upper + lower) / 2

#
#GBM
from sklearn.ensemble import GradientBoostingRegressor

def GBM(q):
    # (a) Modeling
    mod = GradientBoostingRegressor(loss='quantile', alpha=q,
                                    n_estimators=10, max_depth=5,
                                    learning_rate=.01, min_samples_leaf=10,
                                    min_samples_split=10)
    mod.fit(X_train, y_train)
    #mod.fit(X_train['ens_mean'].array.reshape(-1,1), X_train['obs'].array.reshape(-1,1))

    # (b) Predictions
    pred = mod.predict(X_test.array.reshape(1, -1))
    return pred

estimated_quantiles = pd.DataFrame(horizon, columns=['horizon'])
estimated_quantiles[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))

for i in horizon:
    Data_h_i = df_t_2m[(df_t_2m['fcst_hour'] == i)]
    Data_h_i = Data_h_i.reset_index()
    Data_h_i = Data_h_i.drop(columns = ['index', 'init_tm','init_tm_dt','obs_tm','obs_tm_h','met_var', 'ens_sd', 'ens_mean', 'ens_var', 'fcst_hour'])
    X_y_train = Data_h_i[:-1]
    X_y_train = X_y_train.dropna()
    X_test = Data_h_i.drop(columns = ['obs']).iloc[len(Data_h_i)-1]
    y_train = X_y_train['obs']
    X_train = X_y_train.drop(columns = ['obs'])

    for q in quantile_levels:
        estimated_quantiles[str(q)][estimated_quantiles['horizon'] == i] = GBM(q)

rfqr = RandomForestQuantileRegressor(
    random_state=0, min_samples_split=10, n_estimators=1000)
X_train = pd.concat([df_t_2m['ens_mean'][:-1], df_t_2m['obs'][:-1]], axis=1)
X_train = X_train.dropna()
rfqr_fit = rfqr.fit(X_train['ens_mean'].array.reshape(-1,1) , X_train['obs'].values.ravel())
print('a')
rfqr_pred = rfqr.predict(df_t_2m['ens_mean'][len(df_t_2m)-1], quantile=98.5)