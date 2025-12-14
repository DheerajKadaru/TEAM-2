import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('VITE_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')

# List available models
print("Fetching available models...")
url = f'https://generativelanguage.googleapis.com/v1beta/models?key={key}'
r = requests.get(url)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    models = r.json().get('models', [])
    print(f"\nFound {len(models)} models:")
    for m in models[:15]:
        name = m.get('name', '')
        display = m.get('displayName', '')
        print(f"  - {name} ({display})")
    
    # Try the first model that supports generateContent
    for model in models:
        name = model.get('name', '')
        if 'generateContent' in model.get('supportedGenerationMethods', []):
            print(f"\n\nTrying model: {name}")
            test_url = f'https://generativelanguage.googleapis.com/v1beta/{name}:generateContent?key={key}'
            payload = {'contents': [{'role': 'user', 'parts': [{'text': 'Say hello'}]}]}
            test_r = requests.post(test_url, json=payload)
            print(f"Test Status: {test_r.status_code}")
            if test_r.status_code == 200:
                print("✅ SUCCESS! This model works!")
                print(f"Response: {test_r.json()}")
                break
            else:
                print(f"Error: {test_r.text[:200]}")
else:
    print(f"Error: {r.text}")

