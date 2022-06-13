from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
from statsmodels.tsa.arima_model import ARIMAResults
from datetime import datetime, timedelta
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

df = pd.read_csv('data.csv', index_col='Date', parse_dates=True)
df = df.dropna()

loaded = ARIMAResults.load('model.pkl')


@app.route('/', methods=['GET', 'POST'])
def welcome():
    return 'app works'


@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    no_of_days = 7 # for next week

    today = datetime.now()
    n_days = today + timedelta(days=no_of_days)
    print(today.strftime("%Y.%m.%d"), n_days.strftime("%Y.%m.%d"))
    index_future_dates = pd.date_range(start=today.strftime("%Y.%m.%d"), end=n_days.strftime("%Y.%m.%d"))
    # print(index_future_dates)
    pred = loaded.predict(start=len(df), end=len(df) + no_of_days, typ='levels').rename('ARIMA Predictions')
    # print(comp_pred)
    pred.index = index_future_dates
    output_value = []

    for x in range(len(pred)):
        output_value.append(round(pred[x], 2))

    return_str = '{ "result" : ' + str(output_value[0]) + ' }'

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
