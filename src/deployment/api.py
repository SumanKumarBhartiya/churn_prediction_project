from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("models/model.pkl", "rb"))

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"churn_prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
