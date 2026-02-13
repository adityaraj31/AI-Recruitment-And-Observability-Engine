import requests
import json
import os
import time

def test_api():
    # Target the already running server
    base_url = "http://127.0.0.1:8000"
    
    print(f"Connecting to server at {base_url}...")
    
    try:
        # 1. Health Check
        print("Testing /health...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("Health check passed!")
        else:
            print(f"Health check failed: {response.status_code}")
            return

        # 2. Analyze Endpoint with Files (Testing Debate Node)
        print("Testing /analyze with form data (Debate Node flow)...")
        
        # Create dummy temp files
        with open("temp_resume.txt", "w") as f:
            f.write("John Doe. Skills: Python, Docker, LangGraph, AI Agents. Experience: 5 years.")
        with open("temp_jd.txt", "w") as f:
            f.write("Looking for Senior Python Engineer with LangGraph experience and multi-agent systems orchestration.")

        files = {
            'resume_file': ('resume.txt', open('temp_resume.txt', 'rb'), 'text/plain'),
            'jd_file': ('jd.txt', open('temp_jd.txt', 'rb'), 'text/plain')
        }
        
        # Increase timeout because debate node involves 4 LLM calls in total
        start_time = time.time()
        response = requests.post(f"{base_url}/analyze", files=files, timeout=90)
        duration = time.time() - start_time
        
        # Clean up
        files['resume_file'][1].close()
        files['jd_file'][1].close()
        os.remove("temp_resume.txt")
        os.remove("temp_jd.txt")

        if response.status_code == 200:
            print(f"Analyze endpoint passed in {duration:.2f}s!")
            print("DEBATE NODE RESULTS:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Analyze failed: {response.text}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_api()
