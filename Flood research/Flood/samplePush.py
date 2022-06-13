import requests
import json

url = "http://127.0.0.1:2200/push_realtime_data"
data = {'temperature': 20,
        'humidity': 4,
        'pressure': 2,
        'rainfall': 20,
        }

# sending post request and saving response as response object
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json.dumps(data), headers=headers)

# extracting response text
# pastebin_url = r.text
# print("The pastebin URL is:%s" % pastebin_url)
