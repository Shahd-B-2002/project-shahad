
# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
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

# Define and train the SVM model
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, Y_train)

# Make predictions
predictions_svm = model_svm.predict(X_test)

# Display the accuracy and classification report
accuracy = accuracy_score(Y_test, predictions_svm)
print(f"🔹 Accuracy: {accuracy}")
print("🔹 Classification Report:")
print(classification_report(Y_test, predictions_svm))

# Create confusion matrix
conf_matrix = confusion_matrix(Y_test, predictions_svm)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig("result/confusion_matrix_SVM.png")
plt.show()

# Save the predictions to a CSV file
df_preds = pd.DataFrame({'Actual': Y_test.values, 'Predicted': predictions_svm})
df_preds.to_csv("result/predictions_SVM_model.csv", index=False)
