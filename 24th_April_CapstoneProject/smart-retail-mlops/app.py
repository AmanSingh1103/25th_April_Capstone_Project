from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# load model
model = pickle.load(open("model/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Retail Demand API Working"}

@app.post("/predict-demand")
def predict(data: dict):

    # convert input
    df = pd.DataFrame([data])

    # date handling (must match training)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop("Date", axis=1, inplace=True)

    # encoding
    df = pd.get_dummies(df)

    # prediction
    try:
        prediction = model.predict(df)[0]
    except:
        prediction = 0

    return {
        "predicted_demand": round(float(prediction), 2)
    }
