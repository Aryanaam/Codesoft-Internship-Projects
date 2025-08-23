from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Class labels
class_names = ["Setosa", "Versicolor", "Virginica"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
    except:
        return "⚠️ Please enter valid numeric values."

    # Prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Clean static folder (old plots)
    for file in os.listdir("static"):
        if file.endswith(".png"):
            os.remove(os.path.join("static", file))

    # ----- Bar Chart -----
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, probabilities, color=["#FF9999", "#66B2FF", "#99FF99"])
    plt.title("Prediction Probabilities (Bar)")
    plt.xlabel("Flower Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.savefig("static/prediction_bar.png")
    plt.close()

    # ----- Pie Chart -----
    plt.figure(figsize=(5, 5))
    plt.pie(probabilities, labels=class_names, autopct="%1.1f%%",
            colors=["#FF9999", "#66B2FF", "#99FF99"], startangle=140)
    plt.title("Prediction Probabilities (Pie)")
    plt.savefig("static/prediction_pie.png")
    plt.close()

    return render_template("result.html", prediction=prediction)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
