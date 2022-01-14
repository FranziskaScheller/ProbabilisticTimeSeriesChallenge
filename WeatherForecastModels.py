import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.ensemble import GradientBoostingRegressor
from skgarden import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss
from scipy.stats import norm
from datetime import timedelta
from sklearn.linear_model import QuantileRegressor

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
+++ Models for Weather Ensemble Postprocessing +++
"""

"""
EMOS 
"""

def EMOS(q, X_train, y_train, X_test, ind_boosting):

    X_y_train = X_train
    X_y_train['obs'] = y_train

    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_y_train_r = robjects.conversion.py2rpy(X_y_train)
        X_test_r = robjects.conversion.py2rpy(X_test)

    pandas2ri.activate()
    robjects.globalenv['X_y_train_r'] = X_y_train
    if ind_boosting == False:
        robjects.r('''
                   f <- function(X_y_train_r) {
    
                            library(crch)
                            train1.crch <- crch(obs ~ ens_mean_t_2m + ens_skewness_t_2m + ens_mean_clct + ens_mean_mslp|ens_sd_t_2m + ens_sd_clct + ens_sd_mslp, data = X_y_train_r, dist = "gaussian", type = "crps", link.scale = "log")
    
                    }
                    ''')
        # train1.crch <- crch(obs ~ ens_mean_t_2m|ens_sd_t_2m, data = X_y_train_r, dist = "gaussian", type = "crps", link.scale = "log")

    else:
        robjects.r('''
                   f <- function(X_y_train_r) {

                            library(crch)
                            train1.crch <- crch(obs ~ ens_mean_t_2m+ ens_skewness_t_2m + ens_kurtosis_t_2m + ens_iqr_04_06_t_2m + ens_iqr_025_075_t_2m + ens_mean_clct + ens_skewness_clct + ens_kurtosis_clct + ens_iqr_04_06_clct + ens_iqr_025_075_clct + ens_mean_mslp + ens_skewness_mslp + ens_kurtosis_mslp + ens_iqr_04_06_mslp + ens_iqr_025_075_mslp + ens_mean_wind_10m + ens_skewness_wind_10m + ens_kurtosis_wind_10m + ens_iqr_04_06_wind_10m + ens_iqr_025_075_wind_10m  
                            |ens_sd_t_2m + ens_kurtosis_t_2m + ens_iqr_04_06_t_2m + ens_iqr_025_075_t_2m + ens_iqr_01_09_t_2m + ens_sd_clct + ens_kurtosis_clct + ens_iqr_04_06_clct + ens_iqr_025_075_clct + ens_iqr_01_09_clct + ens_sd_mslp + ens_kurtosis_mslp + ens_iqr_04_06_mslp + ens_iqr_025_075_mslp + ens_iqr_01_09_mslp + ens_sd_wind_10m + ens_kurtosis_wind_10m + ens_iqr_04_06_wind_10m + ens_iqr_025_075_wind_10m + ens_iqr_01_09_wind_10m, data = X_y_train_r, dist = "gaussian", type = "crps", link.scale = "log", control = crch.boost(mstop = "aic"))

                    }
                    ''')

    r_f = robjects.globalenv['f']
    rf_model = (r_f(X_y_train_r))
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
    prediction_mu = (r_g(rf_model, X_test_r)).values
    prediction_sd = (r_h(rf_model, X_test_r)).values

    prediction = norm(loc=prediction_mu,
                        scale=prediction_sd).ppf(q)

    return prediction

# define it separately for wind since regressors are different and distribution takes only positive values
def EMOS_wind(q, X_train, y_train, X_test, ind_boosting):
    X_y_train = X_train
    X_y_train['obs'] = y_train

    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_y_train_r = robjects.conversion.py2rpy(X_y_train)
        X_test_r = robjects.conversion.py2rpy(X_test)

    pandas2ri.activate()
    robjects.globalenv['X_y_train_r'] = X_y_train
    if ind_boosting == False:
        robjects.r('''
                   f <- function(X_y_train_r) {

                            library(crch)
                            train1.crch <- crch(obs ~ ens_mean_wind_10m + ens_skewness_t_2m + ens_mean_clct + ens_mean_mslp|ens_sd_wind_10m + ens_sd_clct + ens_sd_mslp, data = X_y_train_r, dist = "gaussian", left = 0, type = "crps", link.scale = "log")

                    }
                    ''')
        # train1.crch <- crch(obs ~ ens_mean_t_2m|ens_sd_t_2m, data = X_y_train_r, dist = "gaussian", type = "crps", link.scale = "log")

    else:
        robjects.r('''
                   f <- function(X_y_train_r) {

                            library(crch)
                            train1.crch <- crch(obs ~ ens_mean_t_2m+ ens_skewness_t_2m + ens_kurtosis_t_2m + ens_iqr_04_06_t_2m + ens_iqr_025_075_t_2m + ens_mean_clct + ens_skewness_clct + ens_kurtosis_clct + ens_iqr_04_06_clct + ens_iqr_025_075_clct + ens_mean_mslp + ens_skewness_mslp + ens_kurtosis_mslp + ens_iqr_04_06_mslp + ens_iqr_025_075_mslp + ens_mean_wind_10m + ens_skewness_wind_10m + ens_kurtosis_wind_10m + ens_iqr_04_06_wind_10m + ens_iqr_025_075_wind_10m  
                            |ens_sd_t_2m + ens_kurtosis_t_2m + ens_iqr_04_06_t_2m + ens_iqr_025_075_t_2m + ens_iqr_01_09_t_2m + ens_sd_clct + ens_kurtosis_clct + ens_iqr_04_06_clct + ens_iqr_025_075_clct + ens_iqr_01_09_clct + ens_sd_mslp + ens_kurtosis_mslp + ens_iqr_04_06_mslp + ens_iqr_025_075_mslp + ens_iqr_01_09_mslp + ens_sd_wind_10m + ens_kurtosis_wind_10m + ens_iqr_04_06_wind_10m + ens_iqr_025_075_wind_10m + ens_iqr_01_09_wind_10m, data = X_y_train_r, dist = "gaussian", left = 0, type = "crps", link.scale = "log", control = crch.boost(mstop = "aic"))

                    }
                    ''')

    r_f = robjects.globalenv['f']
    rf_model = (r_f(X_y_train_r))
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
    prediction_mu = (r_g(rf_model, X_test_r)).values
    prediction_sd = (r_h(rf_model, X_test_r)).values

    prediction = norm(loc=prediction_mu,
                      scale=prediction_sd).ppf(q)

    return prediction

"""
Gradient Boosting with Quantile Loss 
"""
def GBM(q, X_train, y_train, X_test, min_sample_size, n_estimators):
    mod = GradientBoostingRegressor(loss='quantile', alpha=q,
                                    n_estimators=n_estimators, max_depth=8,
                                    learning_rate=.01, min_samples_leaf=min_sample_size,
                                    min_samples_split=15)
    # mod = GradientBoostingRegressor(loss='quantile', alpha=q,
    #                                 n_estimators=10, max_depth=5,
    #                                 learning_rate=.01, min_samples_leaf=10,
    #                                 min_samples_split=10)
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test.array.reshape(1, -1))
    return pred

"""
Quantile Random Forests 
"""
def QRF(q, X_train, y_train, X_test, min_sample_size, n_estimators):
    rfqr = RandomForestQuantileRegressor(
        #random_state=0,
        min_samples_split=min_sample_size, n_estimators=n_estimators)
    rfqr_fit = rfqr.fit(X_train, y_train)
    pred = rfqr.predict(X_test.array.reshape(1, -1), quantile=(q*100))

    return pred


"""
Function for evaluation of quantile forecasts with quantile score
"""

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

def RollingWindowQuantileCalculator(model, data, length_train_data, index_drop_na, horizon, emos_ind_boosting, considered_days, min_sample_size, n_estimator):

    ind = 1

    for h in horizon:
        data_h = data[(data['fcst_hour'] == h)]

        len_data = len(data_h)
        len_preds = len_data - length_train_data

        # Dataframe that contains quantile predictions for the different horizons and test data times
        #quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[length_train_data:len_data-1], columns=['init_tm','obs_tm'])
        quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[length_train_data:len_data],
                                           columns=['init_tm', 'obs_tm'])

        quantile_preds_rw_h['horizon'] = h
        quantile_preds_rw_h[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len(quantile_preds_rw_h), 5))

        quantile_preds_rw_h = quantile_preds_rw_h.reset_index()
        quantile_preds_rw_h = quantile_preds_rw_h.drop(columns='index')

        for i in range(0, len_preds):

            X_y_train = data_h.iloc[i:i+length_train_data]
            X_y_train = X_y_train.dropna()
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
                elif model == EMOS_wind:
                            quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
                            columns=['init_tm', 'obs_tm']), y_train, X_test.drop(
                            columns=['init_tm', 'obs_tm']).to_frame().T, emos_ind_boosting)
                else:
                    quantile_preds_rw_h[str(q)][quantile_preds_rw_h['init_tm'] == time] = model(q, X_train.drop(
                            columns=['init_tm', 'fcst_hour', 'obs_tm']), y_train, X_test, min_sample_size, n_estimator)

        if ind == 1:
                quantile_preds_rw = quantile_preds_rw_h
        else:
                quantile_preds_rw = quantile_preds_rw.append(quantile_preds_rw_h)

        ind = ind + 1

    quantile_preds_rw = quantile_preds_rw.dropna()
    quantile_preds_rw = quantile_preds_rw.reset_index().drop(columns = 'index')

    return quantile_preds_rw

quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]
horizon = [36, 48, 60, 72, 84]

def TotalQuantileCalculator(model, data, horizon, emos_ind_boosting):

    ind = 1

    for h in horizon:
        data_h = data[(data['fcst_hour'] == h)]

        len_data = len(data_h)
        len_preds = 1

        # Dataframe that contains quantile predictions for the different horizons and test data times
        #quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[length_train_data:len_data-1], columns=['init_tm','obs_tm'])
        quantile_preds_rw_h = pd.DataFrame(data_h[['init_tm', 'obs_tm']].iloc[-1:len_data],
                                           columns=['init_tm', 'obs_tm'])

        quantile_preds_rw_h['horizon'] = h
        quantile_preds_rw_h[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((len(quantile_preds_rw_h), 5))

        quantile_preds_rw_h = quantile_preds_rw_h.reset_index()
        quantile_preds_rw_h = quantile_preds_rw_h.drop(columns='index')

        i = 0
        X_y_train = data_h.iloc[i:-1]
        X_y_train = X_y_train.dropna()
        X_y_train = X_y_train.reset_index().drop(columns='index')

        X_train = X_y_train.drop(columns='obs')
        y_train = X_y_train['obs']
        time = data_h['init_tm'].iloc[-1]
        X_test = data_h.drop(columns=['obs', 'init_tm', 'fcst_hour', 'obs_tm']).iloc[-1]

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


def RollingWindowQuantileCalculatorAllModels(weather_data, len_rw_models, index_drop_na, ind_wind):

    quantile_preds_rw_qrf = RollingWindowQuantileCalculator(QRF, weather_data, len_rw_models,
                                                            index_drop_na=index_drop_na, horizon=horizon,
                                                            emos_ind_boosting=False, considered_days=366, min_sample_size = 0, n_estimator = 0)
    quantile_preds_rw_gbm = RollingWindowQuantileCalculator(GBM, weather_data, len_rw_models,
                                                            index_drop_na=index_drop_na, horizon=horizon,
                                                            emos_ind_boosting=False, considered_days=366, min_sample_size = 0, n_estimator = 0)
    if ind_wind == False:
        quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS, weather_data, len_rw_models, index_drop_na=index_drop_na, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = 0, n_estimator = 0)
        quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS, weather_data, len_rw_models, index_drop_na=index_drop_na, horizon=horizon, emos_ind_boosting=True, considered_days = 366, min_sample_size = 0, n_estimator = 0)
    else:
        quantile_preds_rw_emos = RollingWindowQuantileCalculator(EMOS_wind, weather_data, len_rw_models, index_drop_na=index_drop_na, horizon=horizon, emos_ind_boosting=False, considered_days = 366, min_sample_size = 0, n_estimator = 0)
        quantile_preds_rw_emos_boosting = RollingWindowQuantileCalculator(EMOS_wind, weather_data, len_rw_models, index_drop_na=index_drop_na, horizon=horizon, emos_ind_boosting=True, considered_days = 366, min_sample_size = 0, n_estimator = 0)

    quantile_preds_rw_emos = quantile_preds_rw_emos.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_emos_boosting = quantile_preds_rw_emos_boosting.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_qrf = quantile_preds_rw_qrf.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')
    quantile_preds_rw_gbm = quantile_preds_rw_gbm.merge(weather_data[['obs', 'init_tm', 'obs_tm']],
                                                          on=['init_tm', 'obs_tm'], how='left', validate='1:1')

    return quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm


def QRAFitterAndEvaluator(weather_data, len_rw_individual_preds, len_rw_qra_model, ind_wind, alpha_qr):

    if ind_wind == True:
        quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm = RollingWindowQuantileCalculatorAllModels(
            weather_data, len_rw_individual_preds, index_drop_na = True, ind_wind = True)
    else:
        quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm = RollingWindowQuantileCalculatorAllModels(
            weather_data, len_rw_individual_preds, index_drop_na = True, ind_wind = False)
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
        eval_data = eval_data.reset_index()
        eval_data = eval_data.drop(columns='index')

        for i in range(0,len(quantile_preds_rw_qrf_h) - len_rw_qra_model-1):

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
                #X_y_train = X_train
                #X_y_train['obs'] = y_train

                qr = QuantileRegressor(quantile=q, alpha=alpha_qr)
                y_pred = qr.fit(X_train, y_train).predict(X_test)
                #todo: falls das nicht besser wird vllt q's in X weglassen und nochmal probieren ob es an . lag in namen
                # qr = smf.quantreg("obs ~ " + str(q) + 'qrf + ' + str(q) + 'emos + ' + str(q) + 'emos_boosting', X_y_train)
                # res = qr.fit(q=q)
                # y_pred = res.predict(X_test)

                eval_data[str(q)].iloc[i] = y_pred

        if ind == 0:
            eval_data_all_horizons = eval_data
            ind = 1
        else:
            eval_data_all_horizons = eval_data_all_horizons.append(eval_data)

    # prevent quantile crossing
    eval_data_all_horizons[['0.025', '0.25', '0.5', '0.75', '0.975']] = eval_data_all_horizons[
        ['0.025', '0.25', '0.5', '0.75', '0.975']].apply(lambda x: np.sort(x), axis=1, raw=True)

    eval_data_all_horizons = eval_data_all_horizons.dropna()
    eval_data_all_horizons = eval_data_all_horizons.reset_index().drop(columns = 'index')

    return eval_data_all_horizons

def QRAQuantilePredictor(len_rw_individual_preds, weather_data, ind_wind):

    # use all available weather data in the end to fit the last prediction model
    quantile_preds_rw_qrf_all = TotalQuantileCalculator(QRF, weather_data, horizon=horizon, emos_ind_boosting=False)

    if ind_wind == False:
        quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm = RollingWindowQuantileCalculatorAllModels(
            len_rw_individual_preds, index_drop_na=False, ind_wind=False)
        quantile_preds_rw_emos_all = TotalQuantileCalculator(EMOS, weather_data, horizon=horizon, emos_ind_boosting=False)
        quantile_preds_rw_emos_boosting_all = TotalQuantileCalculator(EMOS, weather_data, horizon=horizon, emos_ind_boosting=True)
    else:
        quantile_preds_rw_qrf, quantile_preds_rw_emos, quantile_preds_rw_emos_boosting, quantile_preds_rw_gbm = RollingWindowQuantileCalculatorAllModels(
            len_rw_individual_preds, index_drop_na=False, ind_wind=True)
        quantile_preds_rw_emos_all = TotalQuantileCalculator(EMOS_wind, weather_data, horizon=horizon, emos_ind_boosting=False)
        quantile_preds_rw_emos_boosting_all = TotalQuantileCalculator(EMOS_wind, weather_data, horizon=horizon, emos_ind_boosting=True)

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

        quantile_preds_rw_qrf_all_h = quantile_preds_rw_qrf_all[(quantile_preds_rw_qrf_all['horizon'] == h)]
        quantile_preds_rw_emos_all_h = quantile_preds_rw_emos_all[(quantile_preds_rw_emos_all['horizon'] == h)]
        quantile_preds_rw_emos_boosting_all_h = quantile_preds_rw_emos_boosting_all[
            (quantile_preds_rw_emos_boosting_all['horizon'] == h)]
        #quantile_preds_rw_gbm_all_h = quantile_preds_rw_gbm_all[(quantile_preds_rw_gbm_all['horizon'] == h)]

        eval_data = pd.DataFrame(quantile_preds_rw_qrf_all_h[['init_tm', 'obs_tm', 'horizon']])
        eval_data[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros((1, 5))

        for q in quantile_levels:

            X = pd.DataFrame(quantile_preds_rw_qrf_h[str(q)])
            X = X.rename({str(q): str(q) + 'qrf' }, axis = 'columns')
            X[str(q) + 'emos'] = quantile_preds_rw_emos_h[str(q)]
            X[str(q) + 'emos_boosting'] = quantile_preds_rw_emos_boosting_h[str(q)]
            #X[str(q) + 'gbm'] = quantile_preds_rw_gbm_h[str(q)].iloc[i:i + len_rw_qra_model + 1]
            X_train = X.iloc[0:-1]
            X_test = pd.DataFrame(quantile_preds_rw_qrf_all_h[str(q)])
            X_test = X_test.rename({str(q): str(q) + 'qrf' }, axis = 'columns')
            X_test[str(q) + 'emos'] = quantile_preds_rw_emos_all_h[str(q)]
            X_test[str(q) + 'emos_boosting'] = quantile_preds_rw_emos_boosting_all_h[str(q)]
            y_train = quantile_preds_rw_qrf_h['obs'].iloc[0:-1]

            qr = QuantileRegressor(quantile=q, alpha=0)
            y_pred = qr.fit(X_train, y_train).predict(X_test)
            #todo: falls das nicht besser wird vllt q's in X weglassen und nochmal probieren ob es an . lag in namen
            # qr = smf.quantreg("obs ~ " + str(q) + 'qrf + ' + str(q) + 'emos + ' + str(q) + 'emos_boosting', X_y_train)
            # res = qr.fit(q=q)
            # y_pred = res.predict(X_test)

            eval_data[str(q)].iloc[0] = y_pred

        if ind == 0:
            eval_data_all_horizons = eval_data
            ind = 1
        else:
            eval_data_all_horizons = eval_data_all_horizons.append(eval_data)

    eval_data_all_horizons[['0.025', '0.25', '0.5', '0.75', '0.975']] = eval_data_all_horizons[
        ['0.025', '0.25', '0.5', '0.75', '0.975']].apply(lambda x: np.sort(x), axis=1, raw=True)

    #avg_pinball_loss, avg_pinball_loss_per_quantile, avg_pinball_loss_overall = QuantilePredictionEvaluator(predictions, quantile_levels, horizon)

    return eval_data_all_horizons
