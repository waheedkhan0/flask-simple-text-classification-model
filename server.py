from flask import Flask, request

# Load the model from the file
classifier = joblib.load("sentiment_classifier.joblib")

# Create the Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Get the text input from the request
    text = request.form["text"]

    # Convert the input text to a feature vector
    X = vectorizer.transform([text])

    # Make a prediction
    prediction = classifier.predict(X)

    # Return the prediction
    return prediction[0]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
