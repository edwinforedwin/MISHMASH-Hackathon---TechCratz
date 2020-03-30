#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

test = pd.read_excel(r'C:\Users\Edwin U Kannanaikkal\Documents\MISHMASH Challenge\Test dataset v1.xlsx')
df1 = pd.DataFrame(test)

#Correlation Matrix

corr = df1.corr()
#mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(35,15))
plt.title("Correlation Matrix")
cmap=sns.diverging_palette(240,14, as_cmap='TRUE')
sns.heatmap(corr, cmap=cmap, vmax=0.4,square=True, linewidths=5.0)
