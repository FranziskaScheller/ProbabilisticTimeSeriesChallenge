import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

GDAXI = pd.read_csv('/Users/franziska/Dropbox/DataPTSFC/GDAXI/^GDAXI.csv')
GDAXI = GDAXI.dropna()
GDAXI['Date']= GDAXI['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
GDAXI = GDAXI[GDAXI['Date'] >= datetime.strptime('2020-09-27', '%Y-%m-%d')].reset_index(drop=True)
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

predictions_quant_reg.to_csv('/Users/franziska/Dropbox/DataPTSFC/Submissions/DAX_predictions' + datetime.strftime(datetime.now(), '%Y-%m-%d'), index=False)
"""
Implementation of GARCH model 
Steps: 
1. Check trend and seasonality with ACF (Autocorrelation function), PACF (Partial ACF), Plots and Tests 
"""
rets = rets.dropna()
plt.subplot(511)
plot_acf(rets['ret_1'].values, lags=10)
plt.title('ACF for 1d returns of DAX')
plt.show()

f = plt.figure(figsize=(6, 14))
ax1 = f.add_subplot(511)
rets['ret_1'].plot(ax=ax1)
ax1.title.set_text('Time series of 1d returns')
ax2 = f.add_subplot(512)
rets['ret_2'].plot(ax=ax2)
ax2.title.set_text('Time series of 2d returns')
ax3 = f.add_subplot(513)
rets['ret_3'].plot(ax=ax3)
ax3.title.set_text('Time series of 3d returns')
ax4 = f.add_subplot(514)
rets['ret_4'].plot(ax=ax4)
ax4.title.set_text('Time series of 4d returns')
ax5 = f.add_subplot(515)
rets['ret_5'].plot(ax=ax5)
ax5.title.set_text('Time series of 5d returns')
plt.show()
plt.savefig('raw_DAX_data_returns_different_fc_horizons.png')

f = plt.figure(figsize=(6, 14))
ax1 = f.add_subplot(511)
plot_acf(rets['ret_1'].values, lags=10, ax=ax1)
ax1.title.set_text('ACF of 1d returns')
ax2 = f.add_subplot(512)
plot_acf(rets['ret_2'].values, lags=10, ax=ax2)
ax2.title.set_text('ACF of 2d returns')
ax3 = f.add_subplot(513)
plot_acf(rets['ret_3'].values, lags=10, ax=ax3)
ax3.title.set_text('ACF of 3d returns')
ax4 = f.add_subplot(514)
plot_acf(rets['ret_4'].values, lags=10, ax=ax4)
ax4.title.set_text('ACF of 4d returns')
ax5 = f.add_subplot(515)
plot_acf(rets['ret_5'].values, lags=10, ax=ax5)
ax5.title.set_text('ACF of 5d returns')
plt.show()
plt.savefig('ACF_raw_DAX_data_returns_different_fc_horizons.png')

f = plt.figure(figsize=(6, 14))
ax1 = f.add_subplot(511)
plot_pacf(rets['ret_1'].values, lags=10, ax=ax1)
ax1.title.set_text('PACF of 1d returns')
ax2 = f.add_subplot(512)
plot_pacf(rets['ret_2'].values, lags=10, ax=ax2)
ax2.title.set_text('PACF of 2d returns')
ax3 = f.add_subplot(513)
plot_pacf(rets['ret_3'].values, lags=10, ax=ax3)
ax3.title.set_text('PACF of 3d returns')
ax4 = f.add_subplot(514)
plot_pacf(rets['ret_4'].values, lags=10, ax=ax4)
ax4.title.set_text('PACF of 4d returns')
ax5 = f.add_subplot(515)
plot_pacf(rets['ret_5'].values, lags=10, ax=ax5)
ax5.title.set_text('PACF of 5d returns')
plt.show()
plt.savefig('PACF_raw_DAX_data_returns_different_fc_horizons.png')


basic_gm = arch_model(rets['ret_1'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit()
gm_forecast = gm_result.forecast(horizon=1)
forecast_mean = gm_forecast.mean[-1:]
forecast_var = gm_forecast.variance[-1:]
#gm_result = basic_gm.fit(update_freq = 4)


"""
Select model based on rolling window performance
"""
rets = rets.reset_index(inplace=False)
n = len(rets)
start_date_train = rets['Date'][0]
end_date_train = start_date_train + timedelta(days=365)
train_init = rets[rets['Date'] <= end_date_train]
n_train = len(train_init)

col_names = ["mean_fcst_" + str(i) for i in range(1, 6)] + ["var_fcst_" + str(i) for i in range(1, 6)]
df_forecasts = pd.DataFrame(np.zeros((len(range(0, n-n_train)), 10)), columns=col_names)

for i in range(0, n-n_train):
    train_dat = rets[(start_date_train <= rets['Date']) & (rets['Date'] <= end_date_train)]
    for d in range(1, 6):
        basic_gm = arch_model(train_dat['ret_' + str(d)], p=1, q=1,
                              mean='constant', vol='GARCH', dist='normal')
        # Fit the model
        gm_result = basic_gm.fit()
        gm_forecast = gm_result.forecast(horizon=1)
        df_forecasts.iloc[i][['mean_fcst_' + str(d)]] = gm_forecast.mean[-1:]['h.1'].values
        df_forecasts.iloc[i][['var_fcst_' + str(d)]] = gm_forecast.variance[-1:]['h.1'].values
    
    start_date_train = start_date_train + timedelta(days=1)
    end_date_train = end_date_train + timedelta(days=1)
    
print(1)

#sklearn.metrics.mean_pinball_loss()