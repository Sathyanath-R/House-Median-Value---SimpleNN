import requests

# Define the API endpoint URL
url = "http://localhost:8000/predict"

# Sample input data for prediction
data = {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98
}

# Send POST request to the API endpoint
response = requests.post(url, json=data)

# Check if request was successful
if response.status_code == 200:
    result = response.json()
    print(f"Predicted MEDV value: {result['prediction']}")
else:
    print(f"Failed to get prediction: {response.text}")
