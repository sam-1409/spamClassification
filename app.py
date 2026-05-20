import joblib

print("Loading trained SVM model...")
model = joblib.load("svm_model.pkl")

print("\nSpam Email Classifier")
print("----------------------")

while True:
    email = input("\nEnter email text (or type 'exit' to quit): ")

    if email.lower() == "exit":
        print("Exiting application...")
        break

    prediction = model.predict([email])[0]

    if prediction == "spam":
        print("Prediction: SPAM EMAIL")
    else:
        print("Prediction: NOT SPAM (HAM)")