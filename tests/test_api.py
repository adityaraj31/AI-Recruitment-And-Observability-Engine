import requests
import json
import time
import subprocess
import sys

def test_api():
    # Start the server in a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Waiting for server to start...")
    time.sleep(5)  # Wait for server to boot
    
    try:
        # 1. Health Check
        print("Testing /health...")
        response = requests.get("http://127.0.0.1:8000/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        print("Health check passed!")

        # 2. Analyze Endpoint
        print("Testing /analyze...")
        payload = {
            "resume_text": "John Doe. Skills: Python, Docker.",
            "job_description_text": "Looking for Python Engineer."
        }
        response = requests.post("http://127.0.0.1:8000/analyze", json=payload)
        if response.status_code == 200:
            print("Analyze endpoint passed!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Analyze failed: {response.text}")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        process.terminate()
        print("Server terminated.")

if __name__ == "__main__":
    test_api()
