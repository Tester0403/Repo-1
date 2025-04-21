# 1. Import required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Display summary statistics
print("Summary Statistics:\n", df.describe())
print("\nTarget classes:", iris.target_names)

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree Classifier (using Gini)
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate accuracy and confusion matrix
y_pred = clf.predict(X_test)
print("\nAccuracy (Gini):", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 6. Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree using Gini")
plt.show()

# 7. Predict a custom input
custom_input = [[5.1, 3.5, 1.5, 0.2]]  # Sepal & Petal values
prediction = clf.predict(custom_input)
print("\nCustom Input Prediction:", iris.target_names[prediction[0]])

# 8. Apply pruning/depth control and compare
pruned_clf = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
pruned_clf.fit(X_train, y_train)
y_pruned_pred = pruned_clf.predict(X_test)

print("\nAccuracy (With Depth Control = 2):", accuracy_score(y_test, y_pruned_pred))

# Plot pruned tree
plt.figure(figsize=(10, 6))
plot_tree(pruned_clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Pruned Decision Tree (max_depth=2)")
plt.show()
