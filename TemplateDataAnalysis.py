import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

#%matplotlib inline

# Importing Data set
#from pyodide.http import pyfetch

#async def download(url, filename):
#    response = await pyfetch(url)
#    if response.status == 200:
#        with open(filename, "wb") as f:
#            f.write(await response.bytes())

# File location

#path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"

#you will need to download the dataset;
#await download(path, "laptops.csv")

file_name="laptops.csv"

df = pd.read_csv(file_name, header=0)   #crea dataset
print(df)

#Crea linear Regression 
lm = LinearRegression()
X=df["CPU_frequency"]
Y=df["Price"]

plt.scatter(X,Y)
plt.xlabel("CPU_Frequency")
plt.ylabel("Price")
plt.show()

Yhat=lm.fit(X,Y)
lm.coef_
lm.intercept_

plt.scatter(X,Yhat)
plt.xlabel("CPU_Frequency")
plt.ylabel("Price")
plt.show()



