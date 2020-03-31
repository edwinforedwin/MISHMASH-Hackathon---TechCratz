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

#Plotting Train and valid Data

Train.EQ.plot(figsize=(15,8), title= 'Daily Sales', fontsize=14, label='train', linewidth=0.5) 
Valid.EQ.plot(figsize=(15,8), title= 'Daily Sales', fontsize=14, label='valid', linewidth=0.5)
plt.xlabel("Days") 
plt.ylabel("Sales") 
plt.legend(loc='best') 
plt.show()

