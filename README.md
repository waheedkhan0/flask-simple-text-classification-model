# A Flask App to demonstrate simple text classification model
A simple text classification model and Flask framework to deploy. This was done to learn about how a simple AI solution can be deployed as a service and exposing method to be served.

Here are the steps involved and details:
- Prepare the data: First, you'll need to acquire and prepare a dataset of labeled examples. This dataset should include text inputs and their corresponding labels ("positive" or "negative").

- Train the model: Next, you'll use the dataset to train a text classification model using scikit-learn. This can be done using the CountVectorizer and LogisticRegression classes from scikit-learn.

      from sklearn.feature_extraction.text import CountVectorizer
      from sklearn.linear_model import LogisticRegression

      # Prepare the data
      text_input = ["This is a positive text", "This is a negative text"]
      labels = ["positive", "negative"]

      # Create the CountVectorizer object
      vectorizer = CountVectorizer()

      # Fit the vectorizer to the text input
      X = vectorizer.fit_transform(text_input)

      # Create the LogisticRegression model
      classifier = LogisticRegression()

      # Train the model on the data
      classifier.fit(X, labels)
      
 - Save the trained model: Now that the model is trained, you can save it to a file so that it can be loaded later for making predictions.
 
      import joblib

      # Save the model to a file
      joblib.dump(classifier, "sentiment_classifier.joblib")
      
This saves the model to a file named "sentiment_classifier.joblib" which can be loaded later for making predictions.

- Deploy the model: Now that the model is trained and saved, it can be deployed in a production environment. One way to do this is by creating a web service that can be accessed via an API.
I used Flask, a micro web framework for Python, to create a simple web service that can take a text input and return the predicted sentiment.

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
        
 This code creates a web service that listens for POST requests to the "/predict" endpoint. When it receives a request, it extracts the text input from the request, converts it to a feature vector using the vectorizer object, makes a prediction using the classifier model, and returns the prediction.
 
 - Run the service: Finally, you can run the service by calling the run() method of the Flask app and specifying the host and port that it should listen on.

      if __name__ == "__main__":
          app.run(host="0.0.0.0", port=8000)

This runs a web server on your machine on localhost on port 8000, you can then send post request to it and it will respond with the sentiment.

This is a basic example of how to deploy a machine learning model as a web service using Flask.

## Dependencies
- Flask: A micro web framework for Python that we use to create the web service.
- joblib: A library for saving and loading Python objects, which we use to save and load the trained model.
- numpy: A library for numerical computations that is used by scikit-learn.
- scikit-learn: A library for machine learning that we use to train the text classification model.
- scipy: A library for scientific and technical computations that is also used by scikit-learn.

