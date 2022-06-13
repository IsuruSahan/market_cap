import bme280
import smbus2
from time import sleep
import requests
import json

port = 1
address = 0x76
url = "http://127.0.0.1:2200/push_realtime_data"

bus = smbus2.SMBus(port)
bme280.load_calibration_params(bus, address)

while True:
    bme280_data = bme280.sample(bus, address)
    humidity = bme280_data.humidity
    pressure = bme280_data.pressure
    ambient_temperature = bme280_data.temperature
    print(humidity, pressure, ambient_temperature)
    data = {'temperature': round(ambient_temperature, 2),
            'humidity': round(humidity, 2),
            'pressure': round(pressure, 2),
            'rainfall': 20,
            }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    sleep(5)
    print("weather data sent to api")
