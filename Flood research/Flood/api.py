# import libraries
from flask import Flask, render_template, Response, flash, redirect, url_for, session, request, logging
from flask_cors import CORS
import db
from statsmodels.tsa.arima_model import ARIMAResults
import datetime as DT
import pandas as pd
from datetime import datetime, timedelta
import werkzeug
from collections import Counter
import random
import json
import time

import Weather.FBP

import torch

from Chatbot.model import NeuralNet
from Chatbot.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

today = DT.date.today()

start_date = today + DT.timedelta(days=1)
end_date = today + DT.timedelta(days=7)

loaded = ARIMAResults.load('FloodArea/model.pkl')
df = pd.read_csv('FloodArea/data.csv', index_col='Date', parse_dates=True)
df = df.dropna()

db = db.DB()
current_user = ''

with open('Chatbot/intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "Chatbot/last_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

chat_history = []

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

# Initialize the flask App
app = Flask(__name__)
CORS(app)
app.config['PERMANENT_SESSION_LIFETIME'] = DT.timedelta(minutes=5)

realtime_temperature = 25
realtime_humidity = 20
realtime_pressure = 20
realtime_rainfall = 0


# page links
@app.route('/')
def login():
    global current_user
    current_user = ''
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/home')
def home():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')
    return render_template('home.html')


@app.route('/map')
def map():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')

    no_of_days = 7
    today = datetime.now()
    n_days = today + timedelta(days=no_of_days)
    print(today.strftime("%Y.%m.%d"), n_days.strftime("%Y.%m.%d"))
    index_future_dates = pd.date_range(start=today.strftime("%Y.%m.%d"), end=n_days.strftime("%Y.%m.%d"))
    # print(index_future_dates)
    pred = loaded.predict(start=len(df), end=len(df) + no_of_days, typ='levels').rename('ARIMA Predictions')
    # print(comp_pred)
    pred.index = index_future_dates
    output_value = []
    total = 0
    for x in range(len(pred)):
        total += round(pred[x], 2)
        output_value.append(round(pred[x], 2))

    average = total / 7000;

    return render_template('map.html', average=average)


@app.route('/flood_area', methods=['GET', 'POST'])
def flood_area():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')

    if request.method == 'POST':
        # no_of_days = 7  # for next week
        no_of_days = int(request.form['no_of_days'])
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

        return render_template('flood.html', data=output_value, no_of_days=no_of_days)
    else:
        no_of_days = 7  # for next week
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
        return render_template('flood.html', data=output_value, no_of_days=no_of_days)


@app.route('/realtime_data', methods=['GET', 'POST'])
def realtime_data():
    print('called realtime data *********************************************************')
    realtime_data = str(realtime_temperature) + ',' + str(realtime_humidity) + ',' + str(realtime_pressure) + ',' + str(
        realtime_rainfall)
    return realtime_data


@app.route('/push_realtime_data', methods=['GET', 'POST'])
def push_realtime_data():
    global realtime_temperature
    global realtime_humidity
    global realtime_pressure
    global realtime_rainfall

    temp_data = request.json

    realtime_temperature = temp_data['temperature']
    realtime_humidity = temp_data['humidity']
    realtime_pressure = temp_data['pressure']
    realtime_rainfall = temp_data['rainfall']

    return 'called success'


@app.route('/rain_forecasting', methods=['GET', 'POST'])
def rain_forecasting():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')

    temperature_obj = Weather.FBP.FB('temperature.csv')
    humidity_obj = Weather.FBP.FB('humidity.csv')
    pressure_obj = Weather.FBP.FB('pressure.csv')
    rain_obj = Weather.FBP.FB('rain.csv')

    temperature_values = temperature_obj.get_week()
    humidity_values = humidity_obj.get_week()
    pressure_values = pressure_obj.get_week()
    rain_values = rain_obj.get_week()

    temperature = []
    humidity = []
    pressure = []
    rain = []

    # for realtime data
    data = [realtime_temperature, realtime_humidity, realtime_pressure, realtime_rainfall]

    for i in temperature_values:
        temperature.append(i)
    for i in humidity_values:
        humidity.append(i)
    for i in pressure_values:
        pressure.append(i)
    for i in rain_values:
        rain.append(i)

    return render_template('rain.html', temperature=temperature, humidity=humidity, pressure=pressure, rain=rain,
                           data=data)


@app.route('/chat_bot', methods=['GET', 'POST'])
def chat_bot():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')
    if request.method == 'POST':
        msg = request.form['msg']
        chat_history.append('You : ' + msg)
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    rand = random.choice(intent['responses'])
                    # chat_history.append('Bot : ' + random.choice(intent['responses']))
                    if rand == 'time':
                        chat_history.append('Bot :' + str(time.strftime('%X %x %Z')))
                    elif rand == 'date':
                        chat_history.append('Bot :' + str(time.strftime('%m %d %Y')))
                    elif rand == 'wind_speed':
                        chat_history.append('Bot : wind_speed ' + str(realtime_pressure))
                    elif rand == 'humidity':
                        chat_history.append('Bot : humidity ' + str(realtime_humidity))
                    elif rand == 'pressure':
                        chat_history.append('Bot : pressure ' + str(realtime_pressure))
                    elif rand == 'temperature':
                        chat_history.append('Bot : temperature ' + str(realtime_temperature))
                    elif rand == 'flood':
                        no_of_days = 7
                        today = datetime.now()
                        n_days = today + timedelta(days=no_of_days)
                        print(today.strftime("%Y.%m.%d"), n_days.strftime("%Y.%m.%d"))
                        index_future_dates = pd.date_range(start=today.strftime("%Y.%m.%d"),
                                                           end=n_days.strftime("%Y.%m.%d"))
                        # print(index_future_dates)
                        pred = loaded.predict(start=len(df), end=len(df) + no_of_days, typ='levels').rename(
                            'ARIMA Predictions')
                        # print(comp_pred)
                        pred.index = index_future_dates
                        output_value = []
                        total = 0
                        for x in range(len(pred)):
                            total += round(pred[x], 2)
                            output_value.append(round(pred[x], 2))

                        average = total / 7000;

                        chat_history.append('Bot : average flood level next week ' + str(average))
                    else:
                        chat_history.append('Bot : ' + str(rand))
                    return render_template('chat_bot.html', reply=chat_history)
        else:
            chat_history.append('Bot I do not understand...')
            print(f"{bot_name}: I do not understand...")
    return render_template('chat_bot.html', reply=chat_history)


@app.route('/profile')
def profile():
    user = db.get_user_by_email(email=current_user)

    if len(user) < 1:
        return render_template('login.html')

    return render_template('profile.html', email=user[0].get_email(), name=user[0].get_name(), user_id=user[0].get_id())


# Actions
@app.route('/login_action', methods=['POST'])
def login_action():
    email = request.form['email']
    password = request.form['password']

    if db.login(email=email, password=password):
        global current_user
        current_user = email
        return render_template('home.html')
    else:
        return render_template('login.html', status='Username Or Password Invalid')


@app.route('/register_action', methods=['POST'])
def register_action():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    cpassword = request.form['cpassword']

    if password == cpassword:
        db.save_user(name=name, email=email, password=password)
        return render_template('login.html', status='Register Success')
    else:
        return render_template('register.html', status='Password did not match')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1100, debug=True)
