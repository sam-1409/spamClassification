
# Spam Email Classifier

A Machine Learning based Spam Email Classifier built using Python and Scikit-learn.

This project trains and evaluates:
- Naive Bayes Classifier
- Support Vector Machine (SVM)

The model classifies emails as:
- Spam
- Ham (Not Spam)

---

# Dataset

Dataset File:
`spam_mail_classifier.csv`

Columns:
- `email_text` → Email content
- `label` → spam or ham

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Naive Bayes
- Support Vector Machine (SVM)

---

# Project Structure

```bash
spamClassification/
│
├── spam_mail_classifier.csv
├── train.py
├── app.py
├── requirements.txt
├── README.md
├── naive_bayes_model.pkl
└── svm_model.pkl
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/spamClassification.git
cd spamClassification
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Models

Run:

```bash
python train.py
```

This will:
- Train Naive Bayes Model
- Train SVM Model
- Print Accuracy Scores
- Save trained models

Generated files:
- `naive_bayes_model.pkl`
- `svm_model.pkl`

---

# Run the Spam Classifier

After training:

```bash
python app.py
```

Example:

```bash
Enter email text:
Congratulations! You won a free iPhone.

Prediction:
SPAM EMAIL
```

---

# Machine Learning Workflow

1. Load Dataset
2. Clean and Split Data
3. Convert Text using TF-IDF
4. Train Models
5. Evaluate Accuracy
6. Save Best Model
7. Predict New Emails

---

# Future Improvements

- Add GUI using Streamlit or Flask
- Deploy on Heroku or Render
- Use Deep Learning (LSTM/BERT)
- Add Email API Integration

---
