from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and preprocessing artifacts
model = pickle.load(open("model.pkl", "rb"))
enc = pickle.load(open("encoders.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received:", data)

    # Build dataframe with expected columns (preserve order)
    df = pd.DataFrame([data])
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    # Convert numeric-like columns to numeric (columns without encoders)
    numeric_cols = [c for c in cols if c not in enc]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing numeric values with 0 (or consider other strategies)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Apply label encoders safely for categorical columns
    for col, le in enc.items():
        if col in df.columns:
            # Work on the column values as strings to match training
            vals = df[col].astype(str).fillna("")
            def safe_map(x):
                if x in le.classes_:
                    return int(le.transform([x])[0])
                # unseen label -> use -1 (model should handle or we could map to a default)
                return -1
            df[col] = vals.apply(safe_map)

    # Predict
    pred = model.predict(df)[0]

    # Interpret prediction flexibly (handles 0/1 or 'Yes'/'No')
    positive = False
    try:
        if isinstance(pred, (int, float, np.integer, np.floating)):
            positive = int(pred) == 1
        else:
            s = str(pred).strip().lower()
            positive = s in ("yes", "y", "1", "true", "t")
    except Exception:
        positive = False

    return jsonify({"churn": "Yes" if positive else "No"})


if __name__ == "__main__":
    app.run(debug=True)

