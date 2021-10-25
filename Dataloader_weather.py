import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
from DataPreparer_weather import DataPreparer
from datetime import datetime, timedelta
import dwdweather
from dwdweather import DwdWeather

base = importr('base')
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)
# R package names
packnames = ('ggplot2', 'hexbin')
# R vector of strings
from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


#dwd = DwdWeather(resolution='hourly')
#query_hour = datetime(2014, 3, 22, 12)
#result = dwd.query(station_id='04177', timestamp=query_hour)


# Import data

def DataLoaderHistWeather():
    """
    Function that loads icon_eps_data from RData in DataFrame and saves result in csv file
    No inputs, nothing that is returned
    """
    WEATHER_VARS = ["aswdir_s", "clct", "mslp", "t_2m", "t_850hPa", "vmax_10m", "wind_10m"]
    ENS_COLS = ["ens_" + str(i) for i in range(1, 41)] + ["ens_mean", "ens_var"]

    data = []
    numeric_cols = ENS_COLS + ["obs"]
    for wv in WEATHER_VARS:
        icon_eps = robjects.r["load"]("./data/weather/icon_eps_data/icon_eps_" + wv + ".RData")
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.r['data_icon_eps']
            df = robjects.conversion.rpy2py(r_df)
            df.drop(columns=["location"], inplace=True)

            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            df["fcst_hour"] = df["fcst_hour"].astype(np.int32)

            df["init_tm"] = df["init_tm"].dt.date
            df["obs_tm"] = df["obs_tm"].dt.date

            data.append(df)

    df = pd.concat(data, join='inner')

    df.to_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_R_data.csv', index = False)

    return

def DataUpdaterWeather(update_only_R_data):
    """
    Function for updating icon_eps_weather_data with two functionalities
    1. Updates RData by appending new ensemble forecasts (from October onwards) and saves updated data as csv file twice:
    - once as icon_eps_weather_R_data_updated.csv in order to have one file that contains all data available so far
        as one data source (and also backup data if something gets overwritten) since the older ensemble weather forecasts may be deleted from git repo (function only executed once in beginning of challenge)
    - once as icon_eps_weather_full.csv which will be the file that is regularly appended by new ensemble forecasts
    2. Appends new ensemble forecasts from txt files from git to icon_eps_weather_full.csv
    :param update_only_R_data: Indicator function indicating if only RData is updated (=True) or existing data is appended by new forecasts (=False)
    :return: Nothing (but updated data files are saves as csv files)
    """
    if update_only_R_data == True:
        DataLoaderHistWeather()
        df = pd.read_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_R_data.csv')

        first_date = datetime.strptime(max(df['init_tm'].values), '%Y-%m-%d') + timedelta(days = 1)
        last_date = datetime.strptime(datetime.strftime(datetime.now(), '%Y-%m-%d'), '%Y-%m-%d') - timedelta(days = 1)

        new_weather_forecasts = DataPreparer(datetime.strftime(first_date, '%Y-%m-%d'), datetime.strftime(last_date, '%Y-%m-%d'))

        data_full = df.append(new_weather_forecasts)
        data_full.to_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_R_data_updated.csv', index = False)
        data_full.to_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_full.csv',
                     index=False)

    else:
        data_newest_version = pd.read_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_full.csv')

        first_date = datetime.strptime(max(data_newest_version['init_tm'].values), '%Y-%m-%d') + timedelta(days = 1)
        last_date = datetime.strptime(datetime.strftime(datetime.now(), '%Y-%m-%d'), '%Y-%m-%d') - timedelta(days = 1)

        if (first_date > last_date):
            print('Data already up to date')
        else:
            first_date = first_date.strftime('%Y-%m-%d')
            last_date = last_date.strftime('%Y-%m-%d')
            new_weather_forecasts = DataPreparer(first_date, last_date)
            # file_path = '/Users/franziska/PycharmProjects/PTSFC/data/weather/weather_data_summary' + first_date + last_date
            # new_weather_data = pd.read_csv(file_path)

            data_full = data_newest_version.append(new_weather_forecasts)
            data_full.to_csv('/Users/franziska/PycharmProjects/PTSFC/data/weather/icon_eps_weather_full.csv', index = False)

    return

def DataLoaderWeather(df):

    df_aswdir_s = df[df['met_var'] == 'aswdir_s'].reset_index()
    df_aswdir_s.drop(['index'], axis=1, inplace=True)
    df_clct = df[df['met_var'] == 'clct'].reset_index()
    df_clct.drop(['index'], axis=1, inplace=True)
    df_mslp = df[df['met_var'] == 'mslp'].reset_index()
    df_mslp.drop(['index'], axis=1, inplace=True)
    df_t_2m = df[df['met_var'] == 't_2m'].reset_index()
    df_t_2m.drop(['index'], axis=1, inplace=True)
    df_t_850hPa = df[df['met_var'] == 't_850hPa'].reset_index()
    df_t_850hPa.drop(['index'], axis=1, inplace=True)
    df_vmax_10m = df[df['met_var'] == 'vmax_10m'].reset_index()
    df_vmax_10m.drop(['index'], axis=1, inplace=True)
    df_wind_10m = df[df['met_var'] == 'wind_10m'].reset_index()
    df_wind_10m.drop(['index'], axis=1, inplace=True)

    return df_aswdir_s, df_clct, df_mslp, df_t_2m, df_t_850hPa, df_vmax_10m, df_wind_10m

def RealObservationsAdder(file_path_data_full, file_path_for_update, variable_indicator):
    real_obs = pd.read_csv(file_path_for_update, sep=';')
    real_obs['MESS_DATUM'] = real_obs['MESS_DATUM'].apply(
        lambda x: datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8]), int(str(x)[8:10])))
    data_full = pd.read_csv(file_path_data_full)
    data_full_merge = data_full


    if variable_indicator == 't_2m':
        temperature = real_obs[['MESS_DATUM', 'TT_TU']]
        temperature['met_var'] = 't_2m'
        #data_full[(data_full['met_var'] == 't_2m') & (np.isnan(data_full['obs']) == True)]['obs'] =

        data_full_merge['init_tm_dt'] = data_full_merge['init_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data_full_merge['MESS_DATUM'] = data_full_merge['init_tm_dt'] + pd.to_timedelta(data_full_merge['fcst_hour'],'h')
        data_full_merge = data_full_merge.drop(columns = ['init_tm_dt'])
        data_full_merge = data_full_merge.merge(temperature, on = ['met_var', 'MESS_DATUM'], how = 'outer')
        data_full_merge['obs'] = data_full_merge['obs'].fillna(data_full_merge['TT_TU'])
        data_full_merge = data_full_merge.drop(columns=['TT_TU'])
        data_full_merge.to_csv(file_path_data_full.replace('.csv','') + '_updated_real_obs.csv', index=False)

    return data_full_merge
