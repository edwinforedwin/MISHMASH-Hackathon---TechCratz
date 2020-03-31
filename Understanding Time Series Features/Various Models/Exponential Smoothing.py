import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller 
from matplotlib.pylab import rcParams 
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Booked Attibute.xlsx')
train = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Training-Data-Sets.xlsx')

df_test = pd.DataFrame(test)
df_train = pd.DataFrame(train)

Train = df_train[0:11000]
Valid = df_train[11000:12001]

#Exponential Smoothing

y_hat_avg = Valid.copy() 
fit2 = SimpleExpSmoothing(np.asarray(Train['EQ'])).fit(smoothing_level=0.6,optimized=False) 
y_hat_avg['SES'] = fit2.forecast(len(Valid)) 
plt.figure(figsize=(16,8)) 
plt.plot(Train['EQ'], label='Train') 
plt.plot(Valid['EQ'], label='Valid') 
plt.plot(y_hat_avg['SES'], label='SES') 
plt.legend(loc='best') 
plt.show()

rms = sqrt(mean_squared_error(Valid.EQ, y_hat_avg.SES)) 
print(rms)
