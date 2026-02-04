# For testing Grafana

import requests
import numpy as np
import time

url = "http://localhost:8000/predict"

while True:
    # Send batch of 10 samples
    features = np.random.rand(10, 29).tolist()
    response = requests.post(url, json={"features": features})
    print(f"Status: {response.status_code}, Predictions: {len(response.json()['predictions'])}")
    time.sleep(2)  # Wait 2 seconds between requests