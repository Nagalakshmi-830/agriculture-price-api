from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI(title="Agriculture Price Prediction API")

# Enable CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and models once
df = pd.read_csv("data/cleaned_dataset.csv")
rf_model = joblib.load("models/rf_model.pkl")
label_encoders = joblib.load("models/rf_label_encoder.pkl")

FEATURE_ORDER = [
    "state", "district_name", "market_name", "commodity",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"
]

def encode_inputs(state, district, market, commodity):
    return {
        "state": label_encoders["state"].transform([state])[0],
        "district_name": label_encoders["district_name"].transform([district])[0],
        "market_name": label_encoders["market_name"].transform([market])[0],
        "commodity": label_encoders["commodity"].transform([commodity])[0],
    }

# -------------------- GET APIs --------------------
@app.get("/")
def root():
    return {"message": "Agriculture Price Prediction API is running"}

@app.get("/states")
def get_states():
    return {"states": sorted(df["state"].unique().tolist())}

@app.get("/districts")
def get_districts(state: str):
    districts = df[df["state"] == state]["district_name"].unique().tolist()
    return {"districts": sorted(districts)}

@app.get("/markets")
def get_markets(state: str, district: str):
    markets = df[(df["state"] == state) & (df["district_name"] == district)]["market_name"].unique().tolist()
    return {"markets": sorted(markets)}

@app.get("/commodities")
def get_commodities():
    return {"commodities": sorted(df["commodity"].unique().tolist())}

@app.get("/health")
def health():
    return {"status": "API running"}

# -------------------- PREDICTION API --------------------
@app.post("/predict")
def predict(data: dict):
    state = data["state"]
    district = data["district"]
    market = data["market"]
    commodity = data["commodity"]
    days = int(data["days"])

    enc = encode_inputs(state, district, market, commodity)

    history = df[(df["market_name"] == market) & (df["commodity"] == commodity)].sort_values("price_date")
    current_price = history.iloc[-1]["modal_price"]
    predictions = []

    for day in range(1, days + 1):
        row = {**enc,
               "lag_1": current_price,
               "lag_7": current_price,
               "lag_14": current_price,
               "rolling_mean_7": current_price,
               "rolling_mean_14": current_price}
        X = pd.DataFrame([[row[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
        predicted_price = rf_model.predict(X)[0]
        current_price = predicted_price
        predictions.append({"day": day, "price": round(float(predicted_price), 2)})

    return {"predicted_prices": predictions}