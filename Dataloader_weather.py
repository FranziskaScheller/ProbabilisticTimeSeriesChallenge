import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import kurtosis, iqr, skew

base = importr('base')
utils = rpackages.importr('utils')
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

def DataLoaderHistWeather():
    """
    Function that loads icon_eps_data from RData in DataFrame for variables specified in WEATHER_VARS, saves result in
    csv file and returns dataframe
    :return: df (Dataframe) with icon_eps_Rdata
    """
    WEATHER_VARS = ["aswdir_s", "clct", "mslp", "t_2m", "wind_10m"]
    ENS_COLS = ["ens_" + str(i) for i in range(1, 41)] + ["ens_mean", "ens_var"]

    data = []
    numeric_cols = ENS_COLS + ["obs"]
    for wv in WEATHER_VARS:
        icon_eps = robjects.r["load"]("./Data/Weather/icon_eps_Rdata/icon_eps_" + wv + ".RData")
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.r['data_icon_eps']
            df = robjects.conversion.rpy2py(r_df)
            df.drop(columns=["location"], inplace=True)

            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            df["fcst_hour"] = df["fcst_hour"].astype(np.int32)
            #df["init_tm"] = df["init_tm"].dt.date
            #df["obs_tm"] = df["obs_tm"].dt.date

            data.append(df)

    df = pd.concat(data, join='inner')

    df.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_R_data.csv', index=False)

    return df

def DataUpdaterWeather(update_date, real_obs):

    """
    Function for updating icon_eps_weather_data with new ensemble forecasts from git repository
    :param update_date: (str) Date of ensemble forecasts which are appended to icon_eps_weather_R_data
    :return: data_updated (DataFrame) which contains icon_eps_weather_R_data and ensemble forecasts from update_date
    for all weather variables contained in icon_eps_weather_R_data
    """
    hist_weather_ensemble = DataLoaderHistWeather()
    update_date = datetime.strptime(update_date, '%Y-%m-%d')

    new_weather_forecasts = DataPreparer(update_date)
    new_weather_forecasts['init_tm'] = new_weather_forecasts['init_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    weather_data = hist_weather_ensemble.append(new_weather_forecasts).reset_index().drop(columns=['index'])
    weather_data.to_csv('/Users/franziska/Dropbox/DataPTSFC/icon_eps_weather_full.csv', index = False)

    # for each variable add the ensemble mean and standard deviation
    weather_data['ens_mean'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].mean(axis=1)
    weather_data['ens_sd'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].std(axis=1)
    weather_data['ens_skewness'] = skew(weather_data[["ens_" + str(i) for i in range(1, 41)]], axis=1)
    weather_data['ens_kurtosis'] = kurtosis(weather_data[["ens_" + str(i) for i in range(1, 41)]], axis=1)
    weather_data['ens_iqr_025_075'] = iqr(weather_data[["ens_" + str(i) for i in range(1, 41)]], axis=1)
    weather_data['ens_iqr_01_09'] = iqr(weather_data[["ens_" + str(i) for i in range(1, 41)]], rng=(10, 90), axis=1)
    weather_data['ens_iqr_04_06'] = iqr(weather_data[["ens_" + str(i) for i in range(1, 41)]], rng=(40, 60), axis=1)
    # weather_data['ens_kurtosis'] = weather_data[["ens_" + str(i) for i in range(1, 41)]].apply(lambda x: kurtosis(x))
    # split large weather_data DataFrame in smaller DataFrames for the different weather variables
    df_aswdir_s, df_clct, df_mslp, df_t_2m, df_wind_10m = DataLoaderWeather(weather_data)

    # add month and year as variables since temperature depends heavily on the month of the year
    # and possibly a bit on the year itself due to climate change
    df_t_2m['month'] = df_t_2m['obs_tm'].apply(lambda x: x.to_pydatetime().month)
    df_t_2m['year'] = df_t_2m['obs_tm'].apply(lambda x: x.to_pydatetime().year)

    if real_obs == 'temperature':
        df_t_2m_mod = df_t_2m[
            ['init_tm', 'fcst_hour', 'obs_tm', 'obs', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09', 'month', 'year']]
        df_t_2m_mod = df_t_2m_mod.rename(
            columns={'ens_mean': 'ens_mean_t_2m', 'ens_sd': 'ens_sd_t_2m', 'ens_skewness': 'ens_skewness_t_2m',
                     'ens_kurtosis': 'ens_kurtosis_t_2m', 'ens_iqr_04_06': 'ens_iqr_04_06_t_2m',
                     'ens_iqr_025_075': 'ens_iqr_025_075_t_2m', 'ens_iqr_01_09': 'ens_iqr_01_09_t_2m'})
        df_t_2m_mod = df_t_2m_mod.merge(df_clct[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd','ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_t_2m_mod = df_t_2m_mod.rename(
            columns={'ens_mean': 'ens_mean_clct', 'ens_sd': 'ens_sd_clct', 'ens_skewness': 'ens_skewness_clct',
                     'ens_kurtosis': 'ens_kurtosis_clct', 'ens_iqr_04_06': 'ens_iqr_04_06_clct',
                     'ens_iqr_025_075': 'ens_iqr_025_075_clct', 'ens_iqr_01_09': 'ens_iqr_01_09_clct'})
        df_t_2m_mod = df_t_2m_mod.merge(df_mslp[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_t_2m_mod = df_t_2m_mod.rename(
            columns={'ens_mean': 'ens_mean_mslp', 'ens_sd': 'ens_sd_mslp', 'ens_skewness': 'ens_skewness_mslp',
                     'ens_kurtosis': 'ens_kurtosis_mslp', 'ens_iqr_04_06': 'ens_iqr_04_06_mslp',
                     'ens_iqr_025_075': 'ens_iqr_025_075_mslp', 'ens_iqr_01_09': 'ens_iqr_01_09_mslp'})
        df_t_2m_mod = df_t_2m_mod.merge(df_wind_10m[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_t_2m_mod = df_t_2m_mod.rename(
            columns={'ens_mean': 'ens_mean_wind_10m', 'ens_sd': 'ens_sd_wind_10m', 'ens_skewness': 'ens_skewness_wind_10m',
                     'ens_kurtosis': 'ens_kurtosis_wind_10m', 'ens_iqr_04_06': 'ens_iqr_04_06_wind_10m',
                     'ens_iqr_025_075': 'ens_iqr_025_075_wind_10m', 'ens_iqr_01_09': 'ens_iqr_01_09_wind_10m'})
        weather_data = df_t_2m_mod

    else:
        df_wind_10m_mod = df_t_2m[
            ['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09', 'month', 'year']]
        df_wind_10m_mod = df_wind_10m_mod.rename(
            columns={'ens_mean': 'ens_mean_t_2m', 'ens_sd': 'ens_sd_t_2m', 'ens_skewness': 'ens_skewness_t_2m',
                     'ens_kurtosis': 'ens_kurtosis_t_2m', 'ens_iqr_04_06': 'ens_iqr_04_06_t_2m',
                     'ens_iqr_025_075': 'ens_iqr_025_075_t_2m', 'ens_iqr_01_09': 'ens_iqr_01_09_t_2m'})
        df_wind_10m_mod = df_wind_10m_mod.merge(df_clct[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd','ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_wind_10m_mod = df_wind_10m_mod.rename(
            columns={'ens_mean': 'ens_mean_clct', 'ens_sd': 'ens_sd_clct', 'ens_skewness': 'ens_skewness_clct',
                     'ens_kurtosis': 'ens_kurtosis_clct', 'ens_iqr_04_06': 'ens_iqr_04_06_clct',
                     'ens_iqr_025_075': 'ens_iqr_025_075_clct', 'ens_iqr_01_09': 'ens_iqr_01_09_clct'})
        df_wind_10m_mod = df_wind_10m_mod.merge(df_mslp[['init_tm', 'fcst_hour', 'obs_tm', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_wind_10m_mod = df_wind_10m_mod.rename(
            columns={'ens_mean': 'ens_mean_mslp', 'ens_sd': 'ens_sd_mslp', 'ens_skewness': 'ens_skewness_mslp',
                     'ens_kurtosis': 'ens_kurtosis_mslp', 'ens_iqr_04_06': 'ens_iqr_04_06_mslp',
                     'ens_iqr_025_075': 'ens_iqr_025_075_mslp', 'ens_iqr_01_09': 'ens_iqr_01_09_mslp'})
        df_wind_10m_mod = df_wind_10m_mod.merge(df_wind_10m[['init_tm', 'fcst_hour', 'obs_tm','obs', 'ens_mean', 'ens_sd', 'ens_skewness', 'ens_kurtosis', 'ens_iqr_04_06',
             'ens_iqr_025_075', 'ens_iqr_01_09']],
                                        how='left', on=['init_tm', 'fcst_hour', 'obs_tm'], validate="1:1")
        df_wind_10m_mod = df_wind_10m_mod.rename(
            columns={'ens_mean': 'ens_mean_wind_10m', 'ens_sd': 'ens_sd_wind_10m', 'ens_skewness': 'ens_skewness_wind_10m',
                     'ens_kurtosis': 'ens_kurtosis_wind_10m', 'ens_iqr_04_06': 'ens_iqr_04_06_wind_10m',
                     'ens_iqr_025_075': 'ens_iqr_025_075_wind_10m', 'ens_iqr_01_09': 'ens_iqr_01_09_wind_10m'})

        weather_data = df_wind_10m_mod

    return weather_data, df_t_2m, df_wind_10m

def DataLoaderWeather(df):
    """
    Function that splits the data from a large dataframe df into several smaller data frames based on the met_var column
    :param df: (DataFrame) containing a column 'met_var' with the values 'aswdir_s', 'clct', 'mslp', 't_2m', 'wind_10m'
    :return: 5 DataFrames, where each contains only the data from df for a specific met. variable met_var
    """

    df_aswdir_s = df[df['met_var'] == 'aswdir_s'].reset_index()
    df_aswdir_s.drop(['index'], axis=1, inplace=True)
    df_clct = df[df['met_var'] == 'clct'].reset_index()
    df_clct.drop(['index'], axis=1, inplace=True)
    df_mslp = df[df['met_var'] == 'mslp'].reset_index()
    df_mslp.drop(['index'], axis=1, inplace=True)
    df_t_2m = df[df['met_var'] == 't_2m'].reset_index()
    df_t_2m.drop(['index'], axis=1, inplace=True)
    df_wind_10m = df[df['met_var'] == 'wind_10m'].reset_index()
    df_wind_10m.drop(['index'], axis=1, inplace=True)

    return df_aswdir_s, df_clct, df_mslp, df_t_2m, df_wind_10m

"""
Functions that have been used for Karlsruhe weather data for adding all ensemble forecasts and all real observations 
"""

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

        data_full_merge = data_full_merge.merge(temperature, on = ['met_var', 'MESS_DATUM'], how = 'outer')
        data_full_merge['obs'] = data_full_merge['obs'].fillna(data_full_merge['TT_TU'])
        data_full_merge = data_full_merge.drop(columns=['TT_TU'])
        data_full_merge.to_csv(file_path_data_full.replace('.csv','') + '_updated_real_obs_temp.csv', index=False)

    elif variable_indicator == 'wind_10m':
        wind = real_obs[['MESS_DATUM', '   F']]
        wind['   F'] = wind['   F'] * 3.6
        wind['met_var'] = 'wind_10m'
        data_full_merge = data_full_merge.merge(wind, on = ['met_var', 'MESS_DATUM'], how = 'outer')
        data_full_merge['obs'] = data_full_merge['obs'].fillna(data_full_merge['   F'])
        data_full_merge = data_full_merge.drop(columns=['   F'])
        data_full_merge.to_csv(file_path_data_full.replace('.csv','') + '_updated_real_obs_wind.csv', index=False)

    return data_full_merge

def DataPreparer(last_wednesday):
    # todo: "aswdir_s" not there anymore?
    list_variable_names = ["clct", "mslp", "t_2m", "wind_mean_10m"]
    indicator = 0
    for var_name in list_variable_names:
        file_name = '/Users/franziska/PycharmProjects/ProbabilisticTimeSeriesChallenge/kit-weather-ensemble-point-forecast-berlin-old/icon-eu-eps_' + str(last_wednesday.strftime("%Y%m%d%H")) + '_' + var_name + '_Berlin.txt'
        data = pd.read_csv(file_name, sep=",", header=None)
        data = data[4:]
        data = data[0].str.split('|', 42, expand=True)
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
        data['init_tm'] = last_wednesday
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data['obs_tm'] = np.zeros(shape = (len(data),1)).astype(int)
        data['ens_mean'] = np.zeros(shape = (len(data),1)).astype(int)
        data['ens_var'] = np.zeros(shape = (len(data),1)).astype(int)

        for hour in range(0,len(data)):
            data['obs_tm'][hour] = data['init_tm'][hour] + timedelta(hours = int(data['fcst_hour'][hour]))
            #data['ens_mean'][hour] = data[["ens_" + str(i) for i in range(1, 41)]].iloc[hour].mean()
            #data['ens_var'][hour] = data[["ens_" + str(i) for i in range(1, 41)]].iloc[hour].var()

        data['init_tm'] = data['init_tm'].apply(lambda x: x.strftime('%Y-%m-%d'))

        if indicator == 0:
            data_full = data
            indicator = 1
        else:
            data_full = data_full.append(data)

    #todo: change data type format

    data_full['met_var'] = data_full['met_var'].replace('wind_mean_10m', 'wind_10m')
    data_full['ens_mean'] = data_full[["ens_" + str(i) for i in range(1, 41)]].apply(lambda x: x.mean(), axis=1)
    data_full['ens_var'] = data_full[["ens_" + str(i) for i in range(1, 41)]].apply(lambda x: x.var(), axis=1)

    file_path = '/Users/franziska/Dropbox/DataPTSFC/new_weather_data_summary' + last_wednesday.strftime('%Y-%m-%d')

    data_full.to_csv(file_path, index = False)

    return data_full
