import requests

class ZeroGPTClient:
    def __init__(self, base_url="https://api.zerogpt.com", login=None, password=None):
        self.email = login
        self.password = password
        self.base_url = base_url
        self.token = None
        self.api_key = None

        self.login()
        self.generate_api_key()

    def login(self):
        url = f"{self.base_url}/api/auth/login"
        payload = {"email": self.email, "password": self.password}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        self.token = data["data"]["token"]
        print("Login successful. Access token obtained.")

    def generate_api_key(self):
        url = f"{self.base_url}/api/auth/generateApiKey"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        self.api_key = data["data"]["apiKey"]
        print("API key generated.")

    def analyze_text(self, text):
        url = f"{self.base_url}/api/detect/detectText"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "ApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"input_text": text}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        if response.status_code == 200:
            return response.json()
        else:
            return None
