#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Test dataset v1.xlsx')
df1 = pd.DataFrame(test)

#Sales Vs Social Impressions

pcof = pearsonr(df1['EQ'],df1['Social_Search_Impressions'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(8,3))
plt.title("How social Impressions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Social Impression Count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Social_Search_Impressions'],c='RED')
m, b = np.polyfit(df1['EQ'],df1['Social_Search_Impressions'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Sales Vs Digital Impressions

pcof = pearsonr(df1['EQ'],df1['Digital_Impressions'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Digital Impressions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Digital Impression Count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Digital_Impressions'],c='Green')
m, b = np.polyfit(df1['EQ'],df1['Digital_Impressions'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()
