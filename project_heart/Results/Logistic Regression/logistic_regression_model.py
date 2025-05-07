
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("data/original_data/heart.csv")
X = data.drop("target", axis=1)
Y = data["target"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define and train the Logistic Regression model
model_logreg = LogisticRegression(max_iter=1000)
model_logreg.fit(X_train, Y_train)

# Make predictions
predictions_logreg = model_logreg.predict(X_test)

# Print the accuracy and classification report
print("🔹 Accuracy:", accuracy_score(Y_test, predictions_logreg))
print("\n🔹 Classification Report:
", classification_report(Y_test, predictions_logreg))

# Create confusion matrix
cm = confusion_matrix(Y_test, predictions_logreg)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')

# Save the confusion matrix plot
plt.tight_layout()
plt.savefig("data/result/confusion_logreg.png")
plt.show()
