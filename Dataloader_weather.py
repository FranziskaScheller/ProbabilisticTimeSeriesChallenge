import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
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
        icon_eps = robjects.r["load"]("./Data/Weather/icon_eps_data/icon_eps_" + wv + ".RData")
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

    df.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_R_data.csv', index = False)

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
        df = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_R_data.csv')

        first_date = datetime.strptime(max(df['init_tm'].values), '%Y-%m-%d') + timedelta(days = 1)
        last_date = datetime.strptime(datetime.strftime(datetime.now(), '%Y-%m-%d'), '%Y-%m-%d')
        #last_date = datetime.strptime(datetime.strftime(datetime.now(), '%Y-%m-%d'), '%Y-%m-%d') - timedelta(days = 1)

        new_weather_forecasts = DataPreparer(datetime.strftime(first_date, '%Y-%m-%d'), datetime.strftime(last_date, '%Y-%m-%d'))

        data_full = df.append(new_weather_forecasts)
        data_full.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_R_data_updated.csv', index = False)
        data_full.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv',
                     index=False)

    else:
        data_newest_version = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv')

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
            data_full.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv', index = False)

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

    data_full_merge['init_tm_dt'] = data_full_merge['init_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data_full_merge['MESS_DATUM'] = data_full_merge['init_tm_dt'] + pd.to_timedelta(data_full_merge['fcst_hour'], 'h')
    data_full_merge = data_full_merge.drop(columns=['init_tm_dt'])

    if variable_indicator == 't_2m':
        temperature = real_obs[['MESS_DATUM', 'TT_TU']]
        temperature['met_var'] = 't_2m'
        #data_full[(data_full['met_var'] == 't_2m') & (np.isnan(data_full['obs']) == True)]['obs'] =

        data_full_merge = data_full_merge.merge(temperature, on = ['met_var', 'MESS_DATUM'], how = 'outer')
        data_full_merge['obs'] = data_full_merge['obs'].fillna(data_full_merge['TT_TU'])
        data_full_merge = data_full_merge.drop(columns=['TT_TU'])
        data_full_merge.to_csv(file_path_data_full.replace('.csv','') + '_updated_real_obs_temp.csv', index=False)

    elif variable_indicator == 'wind_10m':
        wind = real_obs[['MESS_DATUM', 'F']]
        wind['F'] = wind['F'] * 3.6
        wind['met_var'] = 'wind_10m'
        data_full_merge = data_full_merge.merge(wind, on = ['met_var', 'MESS_DATUM'], how = 'outer')
        data_full_merge['obs'] = data_full_merge['obs'].fillna(data_full_merge['F'])
        data_full_merge = data_full_merge.drop(columns=['F'])
        data_full_merge.to_csv(file_path_data_full.replace('.csv','') + '_updated_real_obs_wind.csv', index=False)

    return data_full_merge

def DataPreparer(first_date, last_date):
    first_date_dt = datetime.strptime(first_date, '%Y-%m-%d')
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')

    all_dates = pd.date_range(first_date_dt, last_date_dt, freq='1D')
    list_variable_names = ["clct", "mslp", "t_2m", "t_850hPa", "vmax_10m", "wind_mean_10m"]
    indicator = 0
    for var_name in list_variable_names:
        for date in all_dates:
            if ((var_name != "t_850hPa") & (date != datetime.strptime('2021-10-02', '%Y-%m-%d'))):
                file_name = '/Users/franziska/PycharmProjects/ProbabilisticTimeSeriesChallenge/kit-weather-ensemble-point-forecast-karlsruhe/icon-eu-eps_' + str(date.strftime("%Y%m%d%H")) + '_' + var_name + '_Karlsruhe.txt'
                data = pd.read_csv(file_name, sep=",", header=None)
                data = data[4:]
                data = data[0].str.split('|', 42, expand=True)
                # ENS_COLS = ["empty", "fcst_hour"] + ["ens_" + str(i) for i in range(1, 41)] + ["empty"]
                data = data.drop(labels=0, axis=1)
                data = data.drop(labels=42, axis=1)
                data = data.astype(np.float32)
                data = data.rename(
                    columns={1: 'fcst_hour', 2: 'ens_1', 3: 'ens_2', 4: 'ens_3', 5: 'ens_4', 6: 'ens_5', 7: 'ens_6',
                             8: 'ens_7', 9: 'ens_8', 10: 'ens_9', 11: 'ens_10', 12: 'ens_11', 13: 'ens_12', 14: 'ens_13',
                             15: 'ens_14', 16: 'ens_15', 17: 'ens_16', 18: 'ens_17', 19: 'ens_18', 20: 'ens_19',
                             21: 'ens_20', 22: 'ens_21', 23: 'ens_22', 24: 'ens_23', 25: 'ens_24', 26: 'ens_25',
                             27: 'ens_26', 28: 'ens_27', 29: 'ens_28', 30: 'ens_29', 31: 'ens_30', 32: 'ens_31',
                             33: 'ens_32', 34: 'ens_33', 35: 'ens_34', 36: 'ens_35', 37: 'ens_36', 38: 'ens_37',
                             39: 'ens_38', 40: 'ens_39', 41: 'ens_40'})
                data['fcst_hour'] = data['fcst_hour'].astype(int)
                data['met_var'] = var_name
                data['init_tm'] = date
                data = data.reset_index()
                data = data.drop(columns = 'index')
                data['obs_tm'] = np.zeros(shape = (len(data),1)).astype(int)

                for hour in range(0,len(data)):
                    data['obs_tm'][hour] = data['init_tm'][hour] + timedelta(hours = int(data['fcst_hour'][hour]))

                data['obs_tm'] = data['obs_tm'].apply(lambda x: x.strftime('%Y-%m-%d'))
                data['init_tm'] = data['init_tm'].apply(lambda x: x.strftime('%Y-%m-%d'))

                if indicator == 0:
                    data_full = data
                    indicator = 1
                else:
                    data_full = data_full.append(data)

    #todo: change data type format

    data_full['met_var'] = data_full['met_var'].replace('wind_mean_10m', 'wind_10m')

    file_path = '/Users/franziska/Dropbox/DataPTSFC/weather_data_summary' + first_date + last_date

    data_full.to_csv(file_path, index = False)

    return data_full


file_path_data_full = '/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv'
full_weather_data = RealObservationsAdder(
    file_path_data_full,
    '/Users/franziska/Dropbox/DataPTSFC/produkt_tu_stunde_20200423_20211020_04177.txt', 't_2m')


file_path_data_full = '/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv'
RealObservationsAdder(file_path_data_full, '/Users/franziska/Dropbox/DataPTSFC/produkt_ff_stunde_20200423_20211020_04177.txt', 'wind_10m')


print(1)