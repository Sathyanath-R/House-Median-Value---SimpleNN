import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Generate a simple dataset
np.random.seed(42)
n_samples = 1000
positive_texts = [f"I love this product {i}" for i in range(n_samples // 2)]
negative_texts = [f"I hate this product {i}" for i in range(n_samples // 2)]
texts = positive_texts + negative_texts
labels = [1] * (n_samples // 2) + [0] * (n_samples // 2)

# Create a DataFrame
df = pd.DataFrame({'text': texts, 'sentiment': labels})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
accuracy = clf.score(X_test_vectorized, y_test)
print(f"Model accuracy: {accuracy}")

# Save the model and vectorizer
joblib.dump(clf, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer saved successfully.")