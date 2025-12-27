from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

# -------------------- APP SETUP --------------------
app = FastAPI(title="Agriculture Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- LOAD DATA & MODELS --------------------
df = pd.read_csv("data/cleaned_dataset.csv")

rf_model = joblib.load("models/rf_model.pkl")
label_encoders = joblib.load("models/rf_label_encoder.pkl")

FEATURE_ORDER = [
    "state",
    "district_name",
    "market_name",
    "commodity",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14"
]

# -------------------- ENCODER --------------------
def encode_inputs_rf(state, district, market, commodity):
    try:
        return {
            "state": int(label_encoders["state"].transform([state])[0]),
            "district_name": int(label_encoders["district_name"].transform([district])[0]),
            "market_name": int(label_encoders["market_name"].transform([market])[0]),
            "commodity": int(label_encoders["commodity"].transform([commodity])[0]),
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid categorical value: {str(e)}"
        )

# -------------------- GET APIs --------------------
@app.get("/")
def root():
    return {"message": "Agriculture Price Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/states")
def get_states():
    return {"states": sorted(df["state"].dropna().unique().tolist())}

@app.get("/districts")
def get_districts(state: str):
    districts = df[df["state"] == state]["district_name"].dropna().unique().tolist()
    return {"districts": sorted(districts)}

@app.get("/markets")
def get_markets(state: str, district: str):
    markets = df[
        (df["state"] == state) &
        (df["district_name"] == district)
    ]["market_name"].dropna().unique().tolist()
    return {"markets": sorted(markets)}

@app.get("/commodities")
def get_commodities():
    return {"commodities": sorted(df["commodity"].dropna().unique().tolist())}

# -------------------- PREDICT API --------------------
@app.post("/predict")
def predict(data: dict):

    # 1. Validate keys
    required_keys = ["state", "district", "market", "commodity", "days"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing keys: {missing}"
        )

    state = data["state"]
    district = data["district"]
    market = data["market"]
    commodity = data["commodity"]
    days = data["days"]

    # 2. Validate days
    if not isinstance(days, int) or days <= 0:
        raise HTTPException(
            status_code=400,
            detail="'days' must be a positive integer"
        )

    # 3. Encode inputs
    enc = encode_inputs_rf(state, district, market, commodity)

    # 4. Get historical data
    history = df[
        (df["market_name"] == market) &
        (df["commodity"] == commodity)
    ].sort_values("price_date")

    if history.empty:
        raise HTTPException(
            status_code=400,
            detail="No historical data found for this market and commodity"
        )

    # 5. Prediction loop
    current_price = float(history.iloc[-1]["modal_price"])
    predictions = []

    for day in range(1, days + 1):
        row = {
            **enc,
            "lag_1": current_price,
            "lag_7": current_price,
            "lag_14": current_price,
            "rolling_mean_7": current_price,
            "rolling_mean_14": current_price
        }

        X = pd.DataFrame(
            [[row[col] for col in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        predicted_price = float(rf_model.predict(X)[0])
        current_price = predicted_price

        predictions.append({
            "day": day,
            "price": round(predicted_price, 2)
        })

    # 6. Response
    return {
        "state": state,
        "district": district,
        "market": market,
        "commodity": commodity,
        "predicted_prices": predictions
    }