from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("heart_model.pkl")
feature_names = joblib.load("feature_names.pkl")


def generate_recommendations(patient_data):
    recs = []

    if patient_data["cholesterol"] > 240:
        recs.append("Reduce saturated fat and fried foods.")

    if patient_data["resting bp s"] > 140:
        recs.append("Monitor blood pressure regularly and reduce salt intake.")

    if patient_data["fasting blood sugar"] == 1:
        recs.append("Control sugar intake and check blood glucose levels.")

    if patient_data["exercise angina"] == 1:
        recs.append("Consult a doctor before intense exercise.")

    if patient_data["max heart rate"] < 100:
        recs.append("Include light cardio exercises like walking.")

    if patient_data["age"] > 50:
        recs.append("Schedule regular heart checkups.")

    if len(recs) == 0:
        recs.append("Maintain a healthy lifestyle and regular exercise.")

    return recs


def risk_level(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Moderate"
    else:
        return "High"


# Route to serve index.html
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    patient_df = pd.DataFrame([data])
    patient_df = patient_df[feature_names]

    pred = model.predict(patient_df)[0]
    prob = model.predict_proba(patient_df)[0][1] * 100

    prediction = "Heart Disease" if pred == 1 else "No Heart Disease"

    contributions = patient_df.iloc[0] * model.coef_[0]
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    contrib_df["Abs"] = contrib_df["Contribution"].abs()
    contrib_df = contrib_df.sort_values(by="Abs", ascending=False)
    top_contrib = contrib_df.head(3)

    factors = []
    for _, row in top_contrib.iterrows():
        direction = "increased" if row["Contribution"] > 0 else "reduced"
        factors.append(f"{row['Feature']} {direction} the risk")

    recommendations = generate_recommendations(data)

    result = {
        "prediction": prediction,
        "risk_score": round(prob, 2),
        "risk_level": risk_level(prob),
        "top_factors": factors,
        "recommendations": recommendations
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
