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

#Sales Data Distribution

plt.figure(figsize=(14,7))
plt.plot(df_test['EQ'])
plt.title('Sales in test Data set')
plt.xlabel('Period')
plt.ylabel('Sales')
plt.show()

#Sales in Train Dataset

plt.figure(figsize=(16,8))
plt.plot(df_train['EQ'],label='Sales Count')
plt.title('Sales in train Data set')
plt.xlabel('Day')
plt.ylabel('Sales')
plt.show()
