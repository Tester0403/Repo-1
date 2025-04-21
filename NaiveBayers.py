# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
import nltk
import string

# Download stopwords if not already done
nltk.download('stopwords')

# 2. Load the SMS Spam Collection dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 3. Preprocess the text
def preprocess(text):
    text = text.lower()  # lowercase
    tokens = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    tokens = tokens.split()  # tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stopwords
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(preprocess)

# 4. Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Multinomial Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Predict a new message
new_msg = ["Win a free iPhone now!"]
new_msg_clean = [preprocess(msg) for msg in new_msg]
new_msg_vec = vectorizer.transform(new_msg_clean)
prediction = model.predict(new_msg_vec)
print("\nNew Message Prediction:", "Spam" if prediction[0] == 1 else "Ham")

# 9. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
