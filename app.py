from flask import Flask, render_template, request, redirect, url_for
from helper import get_prediction
from logger import logging

app = Flask(__name__)

logging.info("Flask server started")

# ----------------------------
# App state
# ----------------------------
reviews = []
positive = 0
negative = 0


# ----------------------------
# Home route (GET)
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    data = {
        "reviews": reviews,
        "positive": positive,
        "negative": negative
    }
    return render_template("index.html", data=data)


# ----------------------------
# Prediction route (POST)
# ----------------------------
@app.route("/", methods=["POST"])
def predict():
    global positive, negative

    text = request.form.get("text", "").strip()

    if not text:
        return redirect(url_for("index"))

    logging.info(f"Input Text: {text}")

    # ✅ SINGLE correct prediction call
    prediction = get_prediction(text)
    logging.info(f"Prediction: {prediction}")

    if prediction == "negative":
        negative += 1
    else:
        positive += 1

    reviews.insert(0, text)

    return redirect(url_for("index"))


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
