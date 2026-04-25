import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/retail.csv")

# -----------------------------
# 2. DATE FEATURE ENGINEERING (IMPORTANT FIX)
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df.drop("Date", axis=1, inplace=True)

# -----------------------------
# 3. HANDLE CATEGORICAL DATA
# -----------------------------
df = pd.get_dummies(df, columns=["ProductID", "Region"])

# -----------------------------
# 4. SPLIT DATA
# -----------------------------
X = df.drop("UnitsSold", axis=1)
y = df["UnitsSold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. SAVE MODEL SAFELY
# -----------------------------
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Training completed successfully!")
print("📦 Model saved at: model/model.pkl")


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# predictions
y_pred = model.predict(X_test)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("📊 Model Evaluation Results:")
print("MAE:", mae)
print("RMSE:", rmse)

