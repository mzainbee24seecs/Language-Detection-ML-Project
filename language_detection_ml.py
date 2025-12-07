import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. Load the dataset
# ---------------------------
data = pd.read_csv("data/language_dataset.csv")

texts = data['text']
labels = data['language']

# ---------------------------
# 2. Convert text to numbers
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)
y = labels

# ---------------------------
# 3. Use all data for training/testing
# ---------------------------
X_train = X
y_train = y
X_test = X
y_test = y

# ---------------------------
# 4. Train the model
# ---------------------------
clf = MultinomialNB()
clf.fit(X_train, y_train)

# ---------------------------
# 5. Predict and show results
# ---------------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
