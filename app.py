from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("best_model_light.pkl")

@app.route("/")
def home():
    return "OnlineFoods model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1]
    return jsonify({"pred": preds.tolist(), "proba": probs.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
