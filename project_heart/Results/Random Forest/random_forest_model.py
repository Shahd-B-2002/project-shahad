
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)

# Make predictions
predictions_rf = model_rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(Y_test, predictions_rf)
print(f"🔹 Accuracy: {accuracy_rf}")
print("\n🔹 Classification Report:")
print(classification_report(Y_test, predictions_rf))

# Create confusion matrix
cm_rf = confusion_matrix(Y_test, predictions_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# Save the confusion matrix plot
os.makedirs("result", exist_ok=True)
plt.savefig("result/confusion_matrix_RF.png")
plt.show()

# Save predictions to a CSV file
pd.DataFrame(predictions_rf, columns=["Predicted"]).to_csv("result/predictions_RF_model.csv", index=False)
