import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt #Librería para gráficas 

df = pd.read_csv('economics.csv', sep=',') 
#print(df.shape)
print(df.describe())

target_column = ['unemploy'] 
predictors = ['pce']

definitions = {
  'unemploy': 'número de desempleados en miles.',
  'pce': 'gastos de consumo personal, en miles de millones de dólares.',
  'psavert': 'tasa de ahorro personal.',
  'uempmed':'duración media del desempleo, en semanas.'
}

x = df[predictors].values
t = df[target_column].values

print(x.shape, t.shape)

lr = LinearRegression()
lr.fit(x, t)

y = lr.predict(x)

plt.scatter(x, t,  color='black')
plt.plot(x, y, color='red', linewidth=3)
plt.xlabel("{}: {}".format(predictors[0], definitions[predictors[0]]))
plt.ylabel("{}: {}".format(target_column[0], definitions[target_column[0]]))
plt.legend(('modelo','muestras'))
plt.show()


