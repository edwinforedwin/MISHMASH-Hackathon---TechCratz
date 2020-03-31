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

Train = df_trained[0:11000]
Valid = df_trained[11000:12001]

model = VAR(endog=Train)
model_fit = model.fit()

prediction = model_fit.forecast(model_fit.y, steps=len(Valid))

#converting predictions to dataframe
cols=df_train.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,15):
    for i in range(0, len(prediction)):
        pred.iloc[i][j] = prediction[i][j]
#check rmse
for i in cols:
  print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], Valid[i])))

#make final predictions

model = VAR(endog=df_trained)
model_fit = model.fit()

#Predicting for 4 periods = 112 days (assuming 28 days in each periods)

yhat = model_fit.forecast(model_fit.y, steps=112)
out = pd.DataFrame(yhat)
out.to_csv('Predi.csv')
