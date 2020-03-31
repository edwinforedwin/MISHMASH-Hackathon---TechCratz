import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Booked Attibute.xlsx')
train = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Training-Data-Sets.xlsx')
