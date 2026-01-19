
import requests
import time

print("Testing Overpass API connectivity and query format...\n")

endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

query = """
[bbox:55.75,37.61,55.76,37.62];
(
  node["amenity"];
);
out center;

try:
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        params={'data': query_simple},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
