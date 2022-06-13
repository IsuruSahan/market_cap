import pandas as pd
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from tensorflow.keras.callbacks import TensorBoard
import time

# constants used in pre-processing

TEMP_COEF = 100

PRESS_SHIFT = 1000
PRESS_COEF = 100
PRESS_DEFAULT = 1000

TIME_ZERO = pd.Timestamp('1970-01-01 00:00:00')
TIME_DELTA = '1h'

SEQ_LENGTH = 48
PERIOD_TO_PREDICT = 1


# functions for cleaning the data
def preprocess_data(data, val_pct=0.2):
    train_x = []
    train_y = []
    val_x = []
    val_y = []

    pct = data.index[-(int(val_pct * len(data)))]

    print("pct:", pct, "data.index[0]:", data.index[0], "data.index[-1]:", data.index[-1], "len(data):", len(data))

    prev_days_x = deque(maxlen=SEQ_LENGTH)
    prev_days_y = deque(maxlen=SEQ_LENGTH)

    for index, row in zip(data.index, data.values):
        if index > data.index[-2 * PERIOD_TO_PREDICT]:
            break
        prev_days_x.append([])
        prev_days_y.append([])
        for n in range(len(row)):
            if n < len(row) / 2:
                if type(row[n]) is not tuple:
                    prev_days_x[len(prev_days_x) - 1].append(row[n])
                else:
                    prev_days_x[len(prev_days_x) - 1].extend(row[n])
            else:
                if type(row[n]) is not tuple:
                    prev_days_y[len(prev_days_y) - 1].append(row[n])
        #                 else:
        #                     prev_days_y[len(prev_days_y) - 1].extend(row[n])

        if len(prev_days_x) == SEQ_LENGTH:
            #             if (rand.rand() < val_pct) TODO! RANDOM SPLIT
            if index < pct:
                train_x.append(np.array(prev_days_x))
                train_y.append(np.array(prev_days_y))
            else:
                val_x.append(np.array(prev_days_x))
                val_y.append(np.array(prev_days_y))

    return (np.array(train_x), np.array(train_y)), (np.array(val_x), np.array(val_y))


def get_labels(data):
    """ returns the list of distinct labels in given data column """
    labels = list(set(data))
    return labels


def data_to_dicts(labels):
    """ returns pair of data to one-hot and one-hot to data dictionaries """
    data_to_oh = {x: tuple(1 if y == labels.index(x) else 0
                           for y in range(len(labels)))
                  for x in labels}

    oh_to_data = {y: x for x, y in data_to_oh.items()}

    return data_to_oh, oh_to_data


def normalize_temp(temp):
    return [float(t) / TEMP_COEF for t in temp]


def denormalize_temp(temp):
    return [t * TEMP_COEF for t in temp]


def normalize_press(press):
    press = [float(p) for p in press]
    for i in range(len(press)):
        if press[i] == 0:
            press[i] = press[i - 1] if i != 0 else PRESS_DEFAULT

    return [(p - PRESS_SHIFT) / PRESS_COEF for p in press]


def denormalize_press(press):
    return [p * PRESS_COEF + PRESS_SHIFT for p in press]


def normalize_time(times):
    """ converts date-time data column to a UNIX-style int (number of TIME_DELTA steps since TIME_ZERO) """
    times = [pd.Timestamp(time[:-6]) for time in times]
    times = [((time - TIME_ZERO) // pd.Timedelta(TIME_DELTA)) for time in times]
    return times


# def denormalize_time(time):
# TODO


def one_hot_encode(data, data_to_oh):
    return [data_to_oh[d] for d in data]


def one_hot_decode(oh, oh_to_data):
    return [oh_to_data[o] for o in oh]


df = pd.read_csv("weatherHistory.csv",
                 names=['time', 'summary', 'precip', 'temp', 'app_temp', 'humidity', 'wind_speed', 'wind_bearing',
                        'visibility', 'loud_cover', 'pressure', 'daily_summary'], low_memory=False)

df = df.drop([0])
df = df.drop(['app_temp', 'wind_speed', 'wind_bearing', 'visibility', 'loud_cover', 'daily_summary'],
             axis=1)  # TODO add wind_speed and other usefull data

print('yyyyyyyy')
print(df)

df.set_index('time', inplace=True)
df.index = normalize_time(df.index)

df.head()
print(df.columns.values)

summary_labels = get_labels(df['summary'])
print("len(summary_labels):", len(summary_labels))

# our training data contains nans when there is no precipitation
df['precip'] = df['precip'].fillna("clear")
precip_labels = get_labels(df['precip'])
print("len(precip_labels):", len(precip_labels))

# daily_summary_labels = get_labels(df['daily_summary'])
# print("len(daily_summary_labels):", len(daily_summary_labels))


summary_to_oh, oh_to_summary = data_to_dicts(summary_labels)
precip_to_oh, oh_to_precip = data_to_dicts(precip_labels)

# print(summary_to_oh, oh_to_summary, sep='\n\n')
# print(precip_to_oh, oh_to_precip, sep='\n\n')

df['summary'] = one_hot_encode(df['summary'], summary_to_oh)
# df['summary'].head()
df['precip'] = one_hot_encode(df['precip'], precip_to_oh)
# df['precip'].head()

df['temp'] = normalize_temp(df['temp'])
df['pressure'] = normalize_press(df['pressure'])
df['humidity'] = df['humidity'].apply(pd.to_numeric)

print('xxxxxxxxxxxx')
print(df)

# print(denormalize_temp(df['temp'])[:5])
# print(denormalize_press(df['pressure'])[:5])
# print(min(df['temp']), max(df['temp']), '\n', min(df['pressure']), max(df['pressure']))

# sorting data by index
df = df.sort_index()

# we shift values so that each row has a corresponding future row
for col in df.columns:
    df["future_{}".format(col)] = df["{}".format(col)].shift(-PERIOD_TO_PREDICT)

(train_x, train_y), (val_x, val_y) = preprocess_data(df, 0.3)

print('test111')
print(df)
print('test2')
# print(train_x)
# print("length of train x:", len(train_x))
# print("length of train y:", len(train_y))
# print("length of val x:", len(val_x))
# print("length of val y:", len(val_y))
# print("ratio:", len(val_x) / (len(train_x) + len(val_x)))

# new_train_y = np.zeros(shape=(train_y.shape[0], 144))
# for i in range(len(train_y)):
#     new_train_y[i] = train_y[i].ravel()
# new_val_y = np.zeros(shape=(val_y.shape[0], 144))
# for i in range(len(val_y)):
#     new_val_y[i] = val_y[i].ravel()
# train_y = train_y.ravel()

# print(new_y.shape)

# constants used in the model

# LSTM_LAYERS = 1
# LSTM_UNITS = 128
#
# FC_LAYERS = 1
# FC_UNITS = 128
#
# INPUT_DIM = (len(summary_labels) + len(precip_labels) + 3) * SEQ_LENGTH
# # OUTPUT_DIM = len(summary_labels) + len(precip_labels) + 3
# OUTPUT_DIM = 3 * SEQ_LENGTH
#
# NAME = "weather_forecaster_{}".format(int(time.time()))
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
# print("train_x:", train_x.shape, "val_x:", val_x.shape)
# print("train_y:", train_y.shape, "val_y:", val_y.shape)
#
# print('test')
# print(train_x)

# model = Sequential()

# for i in range(LSTM_LAYERS):
#     if i == 0:
#         if i != LSTM_LAYERS - 1:
#             model.add(LSTM(LSTM_UNITS, input_shape=(train_x.shape[1:]), return_sequences=True))
#         else:
#             model.add(LSTM(LSTM_UNITS, input_shape=(train_x.shape[1:]), return_sequences=False))
#     else:
#         if i != LSTM_LAYERS - 1:
#             model.add(LSTM(LSTM_UNITS, return_sequences=True))
#         else:
#             model.add(LSTM(LSTM_UNITS, return_sequences=False))
#
# for i in range(FC_LAYERS):
#     model.add(Dense(FC_UNITS, activation='tanh'))

# model.add(Dense(OUTPUT_DIM, activation='tanh'))
# model.summery()
# model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
# model.fit(train_x, new_train_y, epochs=5, batch_size=32, validation_data=(val_x, new_val_y), callbacks=[tensorboard])
# model.save("model.h5")

# new_model = tf.keras.models.load_model('model.h5')
# new_model.summary()
# yhat = new_model.predict(train_x, verbose=0)
# print(yhat)
