import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    meta = {'brands': [], 'year_min': 2000, 'year_max': 2024, 'miles_max': 300000}
    print(f"Error loading model: {e}. Run train_model.py first.")


@app.route("/")
def home():
    return render_template("index.html", meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."})

    try:
        brand = request.form["brand"]
        year = int(request.form["year"])
        km = float(request.form["km_driven"])

        if km < 0 or year < meta['year_min'] or year > meta['year_max']:
            return jsonify({"error": "Please enter valid values."})

        miles = km * 0.621371
        data = pd.DataFrame([{'brand': brand, 'year': year, 'miles_clean': miles}])

        price_usd = np.exp(model.predict(data)[0])
        price_pkr = int(price_usd * 278)

        return jsonify({
            "price_usd": f"${price_usd:,.0f}",
            "price_pkr": f"PKR {price_pkr:,}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
