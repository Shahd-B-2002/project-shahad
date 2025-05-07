
# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load your training and testing data here (X_train, Y_train, X_test, Y_test)
# Example:
# X_train = ...
# Y_train = ...
# X_test = ...
# Y_test = ...

# Define and train the ANN model
model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model_ann.fit(X_train, Y_train)

# Make predictions
predictions_ann = model_ann.predict(X_test)

# Calculate and print the accuracy of the model
accuracy_ann = accuracy_score(Y_test, predictions_ann)
print(f"🔹 Accuracy: {accuracy_ann}")
print("\n🔹 Classification Report:")
print(classification_report(Y_test, predictions_ann))

# Create confusion matrix
cm_ann = confusion_matrix(Y_test, predictions_ann)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix - ANN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# Create directory to save results if it doesn't exist
os.makedirs("result", exist_ok=True)

# Save the confusion matrix plot
plt.savefig("result/confusion_matrix_ANN.png")
plt.show()

# Save the predictions to a CSV file
pd.DataFrame(predictions_ann, columns=["Predicted"]).to_csv("result/predictions_ANN_model.csv", index=False)
