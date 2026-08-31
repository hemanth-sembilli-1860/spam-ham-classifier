from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pandas as pd
import re


# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Select required columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]


# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


data["message"] = data["message"].apply(clean_text)

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})


# Split data before vectorization
X_train_text, X_test_text, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# Convert text into numerical features
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Test with user input
sentence = input("Enter a message: ")

sentence = clean_text(sentence)

vec = vectorizer.transform([sentence])

prediction = model.predict(vec)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")


# Model evaluation
print("\nModel Performance:")
print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test, y_pred))
print("F1 Score  :", f1_score(y_test, y_pred))

