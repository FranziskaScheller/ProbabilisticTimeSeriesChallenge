import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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


DataPreparer('2021-10-10', '2021-10-21')