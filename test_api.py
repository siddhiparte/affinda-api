import requests
import time

def test_api():
    # Base URL
    base_url = "http://127.0.0.1:5000"
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        # Test health endpoint
        print("\nTesting health endpoint...")
        health_response = requests.get(f"{base_url}/health")
        print(f"Health Status: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
        
        # Test prediction endpoint
        print("\nTesting prediction endpoint...")
        test_data = {
            "skills": ["python", "javascript"]
        }
        
        pred_response = requests.post(f"{base_url}/predict", json=test_data)
        print(f"Prediction Status: {pred_response.status_code}")
        print(f"Response: {pred_response.json()}")
        
    except requests.exceptions.ConnectionError:
        print("Connection failed. Please check if the Flask server is running on http://127.0.0.1:5000")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_api()