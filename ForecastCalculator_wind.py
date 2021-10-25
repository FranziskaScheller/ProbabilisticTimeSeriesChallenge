import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

from Dataloader_weather import DataUpdaterWeather, DataLoaderWeather, RealObservationsAdder
from scipy.stats import norm

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

# install.packages("xfun")
xfun = importr('xfun')
scoringRules = rpackages.importr('scoringRules')
crch = rpackages.importr('crch')
""" load wind data """
#DataUpdaterWeather(update_only_R_data=True)
DataUpdaterWeather(update_only_R_data=False)
file_path_data_full = '/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv'
# full_weather_data = pd.read_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_full.csv')
full_wind_data = RealObservationsAdder(file_path_data_full, '/Users/franziska/Dropbox/DataPTSFC/produkt_ff_stunde_20200423_20211020_04177.txt', 'wind_10m')

df_aswdir_s, df_clct, df_mslp, df_t_2m, df_t_850hPa, df_vmax_10m, df_wind_10m = DataLoaderWeather(full_wind_data)

"""
First visualize real wind observations to get a feeling for the data
"""
logging.info('Starting visualization of wind data')

ind = 1
for year in df_wind_10m['MESS_DATUM'].dt.year.unique():
    plt.plot(df_wind_10m['MESS_DATUM'][df_wind_10m['MESS_DATUM'].dt.year == year],
             df_wind_10m['obs'][df_wind_10m['MESS_DATUM'].dt.year == year])
    plt.xlabel('time')
    plt.ylabel('wind in km/h')
    ind = ind + 1
    plt.show()
    plt.savefig(str(year) + 'wind_timeseries_raw_data.png')

df_wind_10m['ens_mean'] = df_wind_10m[["ens_" + str(i) for i in range(1, 41)]].mean(axis=1)
df_wind_10m['ens_var'] = df_wind_10m[["ens_" + str(i) for i in range(1, 41)]].var(axis=1)
df_wind_10m['ens_sd'] = np.sqrt(df_wind_10m['ens_var'])

horizon = [36, 48, 60, 72, 84]
estimated_params = pd.DataFrame(horizon, columns=['horizon'])
estimated_params['mu'] = np.zeros(len(estimated_params))
estimated_params['sd'] = np.zeros(len(estimated_params))
estimated_params['crps'] = np.zeros(len(estimated_params))
estimated_params[['0.025', '0.25', '0.5', '0.75', '0.975']] = np.zeros(len(estimated_params))
for i in horizon:
    wind_10m_data_fcsth_i = df_wind_10m[(df_wind_10m['fcst_hour'] == i)]
    wind_10m_data_fcsth_i = wind_10m_data_fcsth_i.dropna()

    wind_10m_data_fcsth_i_train = wind_10m_data_fcsth_i[0:len(wind_10m_data_fcsth_i) - 1]
    wind_10m_data_fcsth_i_test = wind_10m_data_fcsth_i.iloc[-1:]

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
    crps_fun = scoringRules.crps
    r_float = robjects.vectors.FloatVector
    y_true_r = r_float(wind_10m_data_fcsth_i_test['obs'])
    mu_r = r_float(prediction_mu)
    sigma_r = r_float(prediction_sd)
    score = scoringRules.crps(y_true_r, mean=mu_r, sd=sigma_r, family="normal")
    #    mean_crps_score = np.array(score).mean()

    estimated_params['mu'][estimated_params['horizon'] == i] = prediction_mu
    estimated_params['sd'][estimated_params['horizon'] == i] = prediction_sd
    estimated_params['crps'][estimated_params['horizon'] == i] = score

    quantile_levels = [0.025,0.25,0.5,0.75,0.975]

    for q in quantile_levels:
        percentile_q = norm(loc=estimated_params['mu'][estimated_params['horizon'] == i], scale=estimated_params['sd'][estimated_params['horizon'] == i]).ppf(q)
        estimated_params[str(q)][estimated_params['horizon'] == i] = percentile_q
    #scipy.stats.norm(loc=prediction_mu, scale=prediction_sd).ppf(0.025)

print(1)