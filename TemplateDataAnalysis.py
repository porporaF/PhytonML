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
from numpy.polynomial.polynomial import polyval
warnings.filterwarnings("ignore", category=UserWarning) 
     
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
X=df[['CPU_frequency']]
Y=df['Price']


plt.scatter(X,Y)
plt.xlabel("CPU_Frequency")
plt.ylabel("Price")
plt.show()

lm.fit(X,Y)
Yhat=lm.predict(X)

lm.coef_
lm.intercept_

ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")

# Create a distribution plot for predicted values
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()
# Regressione a 2 parametri
lm2 = LinearRegression()
Z=df[['CPU_frequency','RAM_GB']]
Y=df['Price']

lm2.fit(Z,Y)
Yhat=lm2.predict(Z)
print(lm2.coef_)
print(lm2.intercept_)

ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")

# Create a distribution plot for predicted values
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

#Polinomiale

X = X.to_numpy().flatten()
f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)
f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)
f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
   
PlotPolly(p1, X, Y, 'CPU_frequency')
plt.show()
PlotPolly(p3, X, Y, 'CPU_frequency')
plt.show()
PlotPolly(p5, X, Y, 'CPU_frequency')
plt.show()

print (p5.coef)

a=polyval(1,p5.coef)
print('a=',a)
