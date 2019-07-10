
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'



# pos data
pos_data = pd.read_csv("./01_ForecastingSales/input_files/cs_pos_data.csv")
pos_data.head(6)

pos_data.info()

# check the min and max dates
pos_data['Activity_Date'].min(), pos_data['Activity_Date'].max()

pos_data.drop('Retailer', axis=1, inplace=True)

pos_data['Style_display_code'].value_counts()
# TOP 5
#SZC1066    2416
#AVR8180    2409
#IIP1324    2383
#BGO1833    2373
#MWZ5072    2369

# getting top 1 style
style1_data = pos_data.loc[pos_data['Style_display_code'].isin(ls_scoring)]

# we have 2+ years data
style1_data['Activity_Date'].min(), style1_data['Activity_Date'].max()

# aggregating the sales
style1_data = style1_data.groupby('Activity_Date')['NetSlsUnts_WTD'].sum().reset_index()
style1_data.head()

# format the datetime to pandas timestamp and set it as index
style1_data.reset_index(inplace=True)
style1_data['Activity_Date'] = pd.to_datetime(style1_data['Activity_Date'])

style1_data = style1_data.set_index('Activity_Date')
style1_data.index

# aggregate to monthly to avoid missing weeks frequency
style1_data_monthly = style1_data['NetSlsUnts_WTD'].resample('MS').sum()

# check the 2016 sales
style1_data['2016':]

# see the trend
style1_data.plot(figsize=(15,6))
plt.show()


# plot the timeseries decomposition - trend, seasonality and noise
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(style1_data_monthly, model='additive')
fig = decomposition.plot()
plt.show()


# ARIMA timeseries model => ARIMA(p, d, q) 
#These three parameters account for seasonality, trend, and noise in data:

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# parameter selection for ARIMA
for param in pdq:
  for param_seasonal in seasonal_pdq:
    try:
      mod = sm.tsa.statespace.SARIMAX(style1_data_monthly, order=param,
                                     seasonal_order=param_seasonal,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
      
      results = mod.fit()

      print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
    
    except:
        continue
    
    # Select the lowest one
#    ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:31.053281019038725
#    ARIMA(0, 1, 0)x(1, 1, 0, 12)12 - AIC:49.178884718096754
#    ARIMA(0, 0, 0)x(1, 1, 0, 12)12 - AIC:73.8245721020067
# BEST PARAM = ARIMA(1, 1, 0)x(1, 1, 0, 12)

# Fitting ARIMA model

final_mod = sm.tsa.statespace.SARIMAX(style1_data_monthly,
                                     order=(1,1,0),
                                     seasonal_order=(1, 1, 0, 12),
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)

results = final_mod.fit()
print(results.summary().tables[1])

# checking if any unusual behaviour
results.plot_diagnostics(figsize=(16, 8))
plt.show()



# Validate the predictions

# Visualize
pred = results.get_prediction(start=pd.to_datetime('2016-03-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = style1_data_monthly['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Style1 Sales')
plt.legend()
plt.show()


# RMSE
sales_forecasted = pred.predicted_mean
sales_truth = style1_data_monthly['2016-03-01':]
mse = ((sales_forecasted - sales_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# ------------------------------------------------------------------------------------
using fbprophet
#--------------

from fbprophet import Prophet

import pystan
import cython

# check Pystan
model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = pystan.StanModel(model_code=model_code)  # this will take a minute
y = model.sampling(n_jobs=1).extract()['y']
y.mean()  # should be close to 0


style1_data = style1_data.rename(columns={'Activity_Date': 'ds', 'NetSlsUnts_WTD': 'y'})
style1_model = Prophet(interval_width=0.95)
style1_model.fit(style1_data)

# future predictions
style1_forecast = style1_model.make_future_dataframe(periods=12, freq='MS')
style1_forecast = style1_model.predict(style1_forecast)

# visualize
plt.figure(figsize=(18, 6))
style1_model.plot(style1_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Style1 Sales')


# check the components
style1_model.plot_components(style1_forecast)

# check the values
style1_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)





