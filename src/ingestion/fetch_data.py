import requests
import pandas as pd

API_URL = "https://datasets-server.huggingface.co/rows?dataset=scikit-learn%2Fchurn-prediction&config=default&split=train&offset=0&length=100"

def fetch_data():
    response = requests.get(API_URL)
    data = response.json()
    df = pd.DataFrame(data["features"])
    df.to_csv("data/raw/customer_data.csv", index=False)

if __name__ == "__main__":
    fetch_data()
