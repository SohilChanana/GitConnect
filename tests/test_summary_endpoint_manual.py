
import requests
import sys
import time
import subprocess
import os

def test_summary_endpoint():
    print("Starting server...")
    # Start the server in the background
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    try:
        # Wait for server to start
        time.sleep(20)
        
        # Test 1: octocat/Hello-World (Testing README detection)
        print("\nTest 1: octocat/Hello-World (Verifying README detection)...")
        response = requests.post(
            "http://localhost:8001/summarize",
            json={
                "repo_url": "https://github.com/octocat/Hello-World",
                "use_grounding": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            limitations = data.get("limitations", [])
            readme_missing = any("README not found" in l for l in limitations)
            
            if not readme_missing:
                print("SUCCESS: README detected correctly (no 'README not found' warning).")
            else:
                print("FAILURE: 'README not found' warning is still present.")
                print(f"Limitations: {limitations}")
        else:
            print(f"FAILURE: Server returned {response.status_code}")
            print(response.text)

        # Test 2: Invalid Repo
        print("\nTest 2: Invalid Repo URL...")
        response = requests.post(
            "http://localhost:8001/summarize",
            json={
                "repo_url": "https://github.com/invalid/repo-that-does-not-exist",
                "use_grounding": False
            }
        )
        if response.status_code == 400:
             print("SUCCESS: Correctly handled invalid repo.")
        else:
             print(f"FAILURE: Expected 400, got {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nStopping server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_summary_endpoint()
