import requests
import json

def test_edna_endpoint():
    url = "http://localhost:8000/analyze-edna"
    
    # Sample DNA sequences (fragments)
    payload = {
        "sequences": [
            "ACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT", # Random compliant
            "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATG",
            "NNNNNNNN" # Test with Ns
        ]
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the backend is running with 'uvicorn backend.main:app --reload'")

if __name__ == "__main__":
    test_edna_endpoint()
