
# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Create a directory to save results if it doesn't exist
os.makedirs("result", exist_ok=True)

# Load the dataset
data = pd.read_csv("data/original_data/heart.csv")
X = data.drop("target", axis=1)
Y = data["target"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define and train the Decision Tree model
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, Y_train)

# Make predictions
predictions_tree = model_tree.predict(X_test)

# Display the accuracy and classification report
accuracy = accuracy_score(Y_test, predictions_tree)
print(f"🔹 Accuracy: {accuracy}")
print("🔹 Classification Report:")
print(classification_report(Y_test, predictions_tree))

# Create confusion matrix
conf_matrix = confusion_matrix(Y_test, predictions_tree)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig("result/confusion_matrix_DecisionTree.png")
plt.show()

# Save the predictions to a CSV file
df_preds = pd.DataFrame({'Actual': Y_test.values, 'Predicted': predictions_tree})
df_preds.to_csv("result/predictions_DecisionTree_model.csv", index=False)
