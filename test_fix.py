import requests
import time

def test_analyze_edna():
    print("Testing /analyze-edna...")
    url = "http://localhost:8000/analyze-edna"
    sequences = [
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGT", 
        "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC",
        "AAAAACCCCCGGGGGTTTTTAAAAACCCCCGGGGGTTTTT"
    ]
    
    payload = {"sequences": sequences}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print("Response Data:", data)
        
        if isinstance(data, list) and len(data) == 3:
            print("SUCCESS: Received list of predictions.")
            for item in data:
                print(f"- {item['predicted_taxa']} ({item['confidence']})")
        else:
            print("FAILURE: Unexpected response format.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_analyze_edna()
