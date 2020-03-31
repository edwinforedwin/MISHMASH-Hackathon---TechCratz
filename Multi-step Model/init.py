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

df_train=pd.DataFrame(train)
df_test=pd.DataFrame(test)

df_trained = df_train.drop(['Day','Social_Search_Impressions','Social_Search_Working_cost','Digital_Working_cost','Print_Impressions.Ads40','Print_Working_Cost.Ads50','OOH_Impressions','OOH_Working_Cost','SOS_pct','Digital_Impressions_pct', 'CCFOT', 'Median_Temp', 'Median_Rainfall','Any_Feat_pct_ACV','Any_Disp_pct_ACV', 'EQ_Base_Price','pct_PromoMarketDollars_Category', 'RPI_Category','Magazine_Impressions_pct', 'TV_GRP', 'Competitor1_RPI','Competitor4_RPI','pct_PromoMarketDollars_Subcategory','RPI_Subcategory'],axis=1)
df_trained.index = df_train.Day

#Checking Seasonality

temp = df_trained.drop(['Digital_Impressions','Competitor2_RPI','Competitor3_RPI','pct_ACV'],axis=1)
coint_johansen(temp,1,1).eig

Train = df_trained[0:11000]
Valid = df_trained[11000:12001]
