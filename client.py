import requests

payload = {
    "input": {"prompt": "Jeflon Zuckergates"},
}
response = requests.post("http://localhost:5000/predictions", json=payload)
#breakpoint()
print(response)
