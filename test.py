import requests

url = "https://d89e-2401-4900-8843-57cc-cdb6-af8-25c0-bb20.ngrok-free.app/translate"
payload = {
    "text": "Hello, i am doing great!",
    "to_lang": "hi"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.json())
