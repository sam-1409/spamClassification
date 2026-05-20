import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam_mail_classifier.csv")

# Features and labels
X = df["email_text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Naive Bayes Model
# -------------------------
nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

print("\n===== Naive Bayes Results =====")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

# Save Naive Bayes model
joblib.dump(nb_model, "naive_bayes_model.pkl")

# -------------------------
# SVM Model
# -------------------------
svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LinearSVC())
])

svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

print("\n===== SVM Results =====")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

# Save SVM model
joblib.dump(svm_model, "svm_model.pkl")

print("\nModels saved successfully!")