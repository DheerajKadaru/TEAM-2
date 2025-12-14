import requests

try:
    print("Triggering lazy load via /test-model...")
    response = requests.post("http://localhost:8000/test-model")
    print(f"Status: {response.status_code}")
except Exception as e:
    print(e)
