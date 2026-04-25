from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "API Working"}

@app.post("/predict-failure")
def predict(data: dict):

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    # SAFE probability handling
    try:
        probability = model.predict_proba(input_df)

        # handle single class issue
        if len(probability[0]) == 1:
            prob = float(probability[0][0])
        else:
            prob = float(probability[0][1])

    except:
        prob = 0.0

    return {
        "failure_prediction": int(prediction),
        "failure_probability": prob
    }