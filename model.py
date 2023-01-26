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
