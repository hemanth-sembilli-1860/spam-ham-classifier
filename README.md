# Spam-Ham Classifier

A machine learning project that classifies text messages as **Spam** or **Ham** using Python and the Multinomial Naive Bayes algorithm.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes

## Dataset

The project uses an SMS spam dataset containing messages labeled as:

- **Ham** – Normal message
- **Spam** – Unwanted or fraudulent message

The dataset is stored in `spam.csv`.

## Machine Learning Workflow

1. Load the dataset using Pandas.
2. Select the required message and label columns.
3. Clean the text using lowercase conversion and regular expressions.
4. Convert `ham` and `spam` labels into numerical values.
5. Split the dataset into training and testing sets.
6. Convert text into numerical features using `CountVectorizer`.
7. Train a `MultinomialNB` classifier.
8. Predict whether a new message is Spam or Ham.
9. Evaluate the model using Accuracy, Precision, Recall, and F1 Score.

## Model

**Multinomial Naive Bayes**

Multinomial Naive Bayes is a commonly used classification algorithm for text-based machine learning problems.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-ham-classifier.git
