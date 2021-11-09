import pandas as pd
import numpy as np
from datetime import datetime

def SubmissionFileGenerator(submission_date, forecasts_DAX, forecasts_temp, forecasts_wind):
    """
    Function that generates the final submission csv file in the required format
    :param submission_date: Date of submission in Format 'YYYY-MM-DD'
    :param forecasts_DAX:
    :param forecasts_temp:
    :param forecasts_wind:
    :return: csv file
    """

    header_submission_csv = ['forecast_date', 'target', 'horizon', 'q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']
    data = pd.DataFrame(np.zeros((15, 8)), columns = header_submission_csv)
    data['forecast_date'] = submission_date
    data['target'][0:5] = 'DAX'
    data['target'][5:10] = 'temperature'
    data['target'][10:15] = 'wind'
    data['horizon'] = ['1 day', '2 day', '5 day', '6 day', '7 day', '36 hour', '48 hour', '60 hour', '72 hour', '84 hour', '36 hour', '48 hour', '60 hour', '72 hour', '84 hour']
    data.iloc[0:5, 3:] = forecasts_DAX
    data.iloc[5:10, 3:] = forecasts_temp
    data.iloc[10:15, 3:] = forecasts_wind
    data.to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/' + submission_date.replace('-','') + '_ChandlerBing.csv', index = False)
    return

submission_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
#forecasts_DAX = pd.DataFrame(np.ones((5,5)))
forecasts_DAX = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/DAX_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'))
forecasts_DAX = forecasts_DAX.drop(columns = ['quantile'])
forecasts_DAX = forecasts_DAX.transpose()
forecasts_temp = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/temp_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'))
forecasts_wind = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/wind_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'))
SubmissionFileGenerator(submission_date, forecasts_DAX, forecasts_temp, forecasts_wind)
