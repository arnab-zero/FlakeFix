# run_inference.py
import json
import requests
from prompt import build_flaky_prompt


def read_java_method(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    java_path = "inputs/test_method.java"  # change this
    java_code = read_java_method(java_path)

    prompt = build_flaky_prompt(java_code)

    # Example payload (edit fields to match your API)
    payload = {
        "model": "YOUR_MODEL_NAME",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    # Example request (edit URL/headers)
    url = "http://localhost:11434/api/chat"  # change this
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    print(resp.text)


if __name__ == "__main__":
    main()
