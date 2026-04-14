"""
CarValue — Flask backend for Linear Regression car price predictor
Run: python app.py
Requires: flask, scikit-learn==1.6.1, pandas, numpy
Install: pip install flask scikit-learn==1.6.1 pandas numpy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linearRegressionModel.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("   Place your .pkl file in the same folder as app.py")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /predict": "Predict car price"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Validate required fields
    required = ["name", "company", "year", "kms_driven", "fuel_type"]
    missing = [f for f in required if f not in data or data[f] == "" or data[f] is None]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        name       = str(data["name"])
        company    = str(data["company"])
        year       = int(data["year"])
        kms_driven = int(data["kms_driven"])
        fuel_type  = str(data["fuel_type"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field value: {e}"}), 400

    # The model pipeline expects a DataFrame-like input
    # with columns: name, company, year, kms_driven, fuel_type
    try:
        import pandas as pd
        input_df = pd.DataFrame([{
            "name": name,
            "company": company,
            "year": year,
            "kms_driven": kms_driven,
            "fuel_type": fuel_type
        }])
        prediction = model.predict(input_df)[0]
        # Clamp to reasonable range (model can predict negatives for old/high-km cars)
        prediction = max(0.0, float(prediction))
        return jsonify({
            "predicted_price": round(prediction, 2),
            "unit": "Lakh INR",
            "input": data
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚗 CarValue backend starting...")
    print("   Frontend: open car_price_predictor.html in your browser")
    print("   API:      http://localhost:5000/predict\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
