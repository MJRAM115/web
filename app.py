from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load saved model, scaler, and columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    time = float(request.form.get("time", 0))
    amount = float(request.form.get("amount", 0))
    v1 = float(request.form.get("v1", 0))
    v2 = float(request.form.get("v2", 0))
    v3 = float(request.form.get("v3", 0))
    v4 = float(request.form.get("v4", 0))
    v14 = float(request.form.get("v14", 0))
    v17 = float(request.form.get("v17", 0))

    # Build full input row with zeros for missing columns
    input_dict = {col: 0.0 for col in columns}
    input_dict["Time"] = time
    input_dict["Amount"] = amount
    input_dict["V1"] = v1
    input_dict["V2"] = v2
    input_dict["V3"] = v3
    input_dict["V4"] = v4
    input_dict["V14"] = v14
    input_dict["V17"] = v17

    input_array = np.array([list(input_dict.values())])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        result = "⚠️ Fraud Detected!"
    else:
        result = "✅ Transaction is Safe"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)