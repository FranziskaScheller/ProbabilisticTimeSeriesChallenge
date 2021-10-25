import pandas as pd
import numpy as np
from datetime import datetime

GDAXI = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/GDAXI/^GDAXI.csv')
GDAXI['Date']= GDAXI['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
GDAXI_adj_close = GDAXI[['Date', 'Adj Close']]

def ReturnComputer(y, type, diff_in_periods):

    n = len(y)
    y_2 = y[0+diff_in_periods:].reset_index().drop(columns = 'index')
    y_1 = y[0:n-diff_in_periods].reset_index().drop(columns = 'index')

    if type == 'log':
        ret = 100*(np.log(y_2) - np.log(y_1))
    else:
        ret = 100*((y_2 - y_1)/y_1)

    return ret

ret_1 = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 1)
ret_2 = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 2)
ret_3 = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 3)
ret_4 = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 4)
ret_5 = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 5)

quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

predictions_quant_reg = pd.DataFrame(np.zeros((6,5)), columns=['quantile', '1', '2', '3', '4', '5'])

print(1)