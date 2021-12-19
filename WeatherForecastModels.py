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

"""
Quantile Random Forests 
"""
def QRF(q, X_train, y_train, X_test):
    rfqr = RandomForestQuantileRegressor(
        #random_state=0,
        min_samples_split=5, n_estimators=100)
    rfqr_fit = rfqr.fit(X_train, y_train)
    pred = rfqr.predict(X_test.array.reshape(1, -1), quantile=(q*100))

    return pred
