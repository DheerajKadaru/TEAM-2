import requests
import json
import time

url = "http://localhost:8000/train-model"
data = {
    "sequences": [
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGT",
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGA", 
        "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC", 
        "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCA"
    ],
    "labels": ["Species_A", "Species_A", "Species_B", "Species_B"]
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=data, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print("Response text:", response.text)
    else:
        print(response.json())
except Exception as e:
    print(f"Error: {e}")
