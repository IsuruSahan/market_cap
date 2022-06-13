import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv', index_col='Date', parse_dates=True)
df = df.dropna()
print('Shape of data', df.shape)
print(df.head())


def adf_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


adf_test(df['level'])

stepwise_fit = auto_arima(df['level'], suppress_warnings=True)

print(stepwise_fit.summary())
print(df.shape)
train = df.iloc[:-30]
test = df.iloc[-30:]
print(train.shape, test.shape)
print(test.iloc[0], test.iloc[-1])

model = ARIMA(train['level'], order=(1, 0, 0))
model = model.fit()
model.summary()

start = len(train)
end = len(train) + len(test) - 1

pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA predictions')
# pred.index=index_future_dates
pred.plot(legend=True)
test['level'].plot(legend=True)

rmse = sqrt(mean_squared_error(pred, test['level']))
print(rmse)

model2 = ARIMA(df['level'], order=(2, 1, 1))
model2 = model2.fit()
df.tail()

# model2.save('model.pkl')
