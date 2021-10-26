import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

    nan_array = np.empty(diff_in_periods)
    nan_array[:] = np.nan
    ret_full = pd.DataFrame(nan_array, columns = ['Adj Close'])
    ret_full = ret_full.append(ret, ignore_index=True)

    return ret_full

rets = pd.DataFrame(GDAXI['Date'])
rets['ret_1'] = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 1)['Adj Close']
rets['ret_2'] = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 2)['Adj Close']
rets['ret_3'] = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 3)['Adj Close']
rets['ret_4'] = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 4)['Adj Close']
rets['ret_5'] = ReturnComputer(pd.DataFrame(GDAXI_adj_close['Adj Close']), 'log', 5)['Adj Close']

quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

predictions_quant_reg = pd.DataFrame(np.zeros((5, 6)), columns=['quantile', '1', '2', '3', '4', '5'])
predictions_quant_reg['quantile'] = [str(i) for i in quantile_levels]

for h in range(1,6):
    if h > 1:
        cols = ['ret_' + str(i) for i in (1,h)]
        df_quantreg = rets[cols]
    else:
        df_quantreg = rets[['ret_1']]

    df_quantreg['lag_abs_ret'] = abs(rets['ret_1'].shift(h))
    df_quantreg_test = pd.DataFrame(abs(rets['ret_1'][len(rets)-h:len(rets)-h+1])).rename(columns={"ret_1": "lag_abs_ret"}).reset_index(drop=True)
    i = 0
    for q in quantile_levels:
        qreg = smf.quantreg("ret_" + str(h) + " ~ lag_abs_ret", df_quantreg[:])
        res = qreg.fit(q=q)
        pred = res.predict(df_quantreg_test)
        predictions_quant_reg[str(h)].iloc[i] = pred
        i = i + 1

