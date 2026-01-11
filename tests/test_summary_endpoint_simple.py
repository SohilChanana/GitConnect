
import requests

def test_summary_endpoint():
    print("Test 1: octocat/Hello-World (Verifying README detection)...")
    try:
        response = requests.post(
            "http://localhost:8002/summarize",
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
                print(f"Summary: {data.get('summary_paragraph')[:100]}...")
            else:
                print("FAILURE: 'README not found' warning is still present.")
                print(f"Limitations: {limitations}")
        else:
            print(f"FAILURE: Server returned {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_summary_endpoint()
