#Importing Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

#Reading Dataset

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Test dataset v1.xlsx')
df1 = pd.DataFrame(test)
print(df1.columns)
print(df1.shape)

#Sales Distribution over periods

plt.figure(figsize=(14,8))
plt.grid(which='major',linewidth=0.5,color='green')
plt.plot(df1['EQ'],color='Red')
plt.title("EQ Distribution for each period")
plt.xlabel('Period')
plt.ylabel('Target')
plt.show()
