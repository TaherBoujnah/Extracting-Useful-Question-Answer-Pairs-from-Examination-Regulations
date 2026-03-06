import urllib.request
import json

BASES = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://localhost:11435",
    "http://127.0.0.1:11435",
]

def try_get(url: str):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8")

for base in BASES:
    url = base.rstrip("/") + "/api/tags"
    try:
        status, body = try_get(url)
        print("\n✅ Found Ollama tags endpoint at:", url)
        data = json.loads(body)
        models = [m.get("name") for m in data.get("models", [])]
        print("Models:", models[:20])
        break
    except Exception as e:
        print("❌ Not working:", url, "->", e)
