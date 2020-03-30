#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Test dataset v1.xlsx')
df1 = pd.DataFrame(test)

#Social Impressions

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

#Digital Impressions

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

#Print Impressions

pcof = pearsonr(df1['EQ'],df1['Print_Impressions.Ads40'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Print Impressions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Print Impression Count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Print_Impressions.Ads40'],c='RED',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Print_Impressions.Ads40'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#OOH Impressions

pcof = pearsonr(df1['EQ'],df1['OOH_Impressions'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How OOH_Impressions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("OOH_Impressions Count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['OOH_Impressions'],c='y',marker='D')
m, b = np.polyfit(df1['EQ'],df1['OOH_Impressions'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#SOS_pct 

pcof = pearsonr(df1['EQ'],df1['SOS_pct'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How SOS_pct effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Share Spend")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['SOS_pct'],c='b',marker='D')
m, b = np.polyfit(df1['EQ'],df1['SOS_pct'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Digital Impression pct

pcof = pearsonr(df1['EQ'],df1['Digital_Impressions_pct'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Digital_Impressions_pct effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Digital_Impressions_pct Count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Digital_Impressions_pct'],c='y',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Digital_Impressions_pct'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Stock reinstallation

pcof = pearsonr(df1['EQ'],df1['CCFOT'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How stock reinstallationn effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Stock Reinstatited")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['CCFOT'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['CCFOT'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Temprature

pcof = pearsonr(df1['EQ'],df1['Median_Temp'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Temprature effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Temperature")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Median_Temp'],c='orange',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Median_Temp'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Rainfall

pcof = pearsonr(df1['EQ'],df1['Median_Rainfall'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Rainfall effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Median Rainfall")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Median_Rainfall'],c='black',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Median_Rainfall'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Fuel Price

pcof = pearsonr(df1['EQ'],df1['Fuel_Price'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Fuel effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Fuel Price")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Fuel_Price'],c='r',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Fuel_Price'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Inflation

pcof = pearsonr(df1['EQ'],df1['Inflation'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Trade inflation effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Inflation")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Inflation'],c='magenta',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Inflation'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Trade Invest

pcof = pearsonr(df1['EQ'],df1['Trade_Invest'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Trade Invest effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Trade Invest")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Trade_Invest'],c='y',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Trade_Invest'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Brand Equity

pcof = pearsonr(df1['EQ'],df1['Brand_Equity'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Brand Equity effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Brand Equity")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Brand_Equity'],c='b',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Brand_Equity'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Average Base Price

pcof = pearsonr(df1['EQ'],df1['Avg_EQ_Price'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Avg Base price effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Avg Base Price")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Avg_EQ_Price'],c='black',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Avg_EQ_Price'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Any_Promo_pct_ACV

pcof = pearsonr(df1['EQ'],df1['Any_Promo_pct_ACV'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Store Size effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("store size")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Any_Promo_pct_ACV'],c='violet',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Any_Promo_pct_ACV'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Product Feature

pcof = pearsonr(df1['EQ'],df1['Any_Feat_pct_ACV'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Product Feature effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Product Feature Value")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Any_Feat_pct_ACV'],c='y',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Any_Feat_pct_ACV'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Display Boards

pcof = pearsonr(df1['EQ'],df1['Any_Disp_pct_ACV'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Display board Ads effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Display Board Ad count")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Any_Disp_pct_ACV'],c='r',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Any_Disp_pct_ACV'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Base Price

pcof = pearsonr(df1['EQ'],df1['EQ_Base_Price'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Base Price effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Base Price")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['EQ_Base_Price'],c='b',marker='D')
m, b = np.polyfit(df1['EQ'],df1['EQ_Base_Price'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Estimated Selling

pcof = pearsonr(df1['EQ'],df1['Est_ACV_Selling'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Estimated Selling effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Estimated Selling")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Est_ACV_Selling'],c='black',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Est_ACV_Selling'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Total Sales

pcof = pearsonr(df1['EQ'],df1['pct_ACV'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Total Sales effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Total Sales")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['pct_ACV'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['pct_ACV'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Product Item Count

pcof = pearsonr(df1['EQ'],df1['Avg_no_of_Items'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Items Count effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Avg No of Items")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Avg_no_of_Items'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Avg_no_of_Items'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Sold Item in Market

pcof = pearsonr(df1['EQ'],df1['pct_PromoMarketDollars_Category'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Sold Items effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Items Sold")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['pct_PromoMarketDollars_Category'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['pct_PromoMarketDollars_Category'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Retail Price Index Category

pcof = pearsonr(df1['EQ'],df1['RPI_Category'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Retail Price Index effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Retail Price Index")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['RPI_Category'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['RPI_Category'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Magazine Impressions

pcof = pearsonr(df1['EQ'],df1['Magazine_Impressions_pct'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Magazine Impressions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Magazine Impressions")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Magazine_Impressions_pct'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Magazine_Impressions_pct'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#TV Gross Impressions

pcof = pearsonr(df1['EQ'],df1['TV_GRP'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How TV Gross Impresions effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("TV Gross Impressions")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['TV_GRP'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['TV_GRP'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Competitor 1

pcof = pearsonr(df1['EQ'],df1['Competitor1_RPI'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Competitior1 effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Comp1 RPI")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Competitor1_RPI'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Competitor1_RPI'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Competitor 2

pcof = pearsonr(df1['EQ'],df1['Competitor2_RPI'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Competitior2 effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Comp2 RPI")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Competitor2_RPI'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Competitor2_RPI'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Competitor 3

pcof = pearsonr(df1['EQ'],df1['Competitor3_RPI'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Competitior3 effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Comp3 RPI")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Competitor3_RPI'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Competitor3_RPI'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Competitor 4

pcof = pearsonr(df1['EQ'],df1['Competitor4_RPI'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Competitior4 effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Comp4 RPI")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['Competitor4_RPI'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['Competitor4_RPI'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Sales Category

pcof = pearsonr(df1['EQ'],df1['EQ_Category'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Sales Catagory effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Sales Catagory")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['EQ_Category'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['EQ_Category'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Sales sub category

pcof = pearsonr(df1['EQ'],df1['EQ_Subcategory'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Sales Sub Catagory effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Sales Sub Catagory")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['EQ_Subcategory'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['EQ_Subcategory'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Promotions made on a sub category

pcof = pearsonr(df1['EQ'],df1['pct_PromoMarketDollars_Subcategory'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How Promotions on a Sub Catagory effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("Dollers spend for promotion on Sub Catagory")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['pct_PromoMarketDollars_Subcategory'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['pct_PromoMarketDollars_Subcategory'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

#Retail Price Index on Sub Category

pcof = pearsonr(df1['EQ'],df1['RPI_Subcategory'])
print("Correlation Coefficent and p value : \n \n",pcof)
plt.figure(figsize=(10,4))
plt.title("How RPI of Sub Catagory effects EQ")
plt.grid(which='major',linewidth=0.5,color='green')
plt.ylabel("RPI of Sub Catagory")
plt.xlabel("Periods")
plt.scatter(df1['EQ'],df1['RPI_Subcategory'],c='m',marker='D')
m, b = np.polyfit(df1['EQ'],df1['RPI_Subcategory'], 1)
plt.plot(df1['EQ'], m*df1["EQ"] + b)
plt.show()

