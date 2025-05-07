
# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd

# Create a directory to save results if it doesn't exist
os.makedirs("result", exist_ok=True)

# Define and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

# Make predictions
Y_pred_nb = nb_model.predict(X_test)

# Calculate and print the accuracy
accuracy = nb_model.score(X_test, Y_test)
print("🔹 Accuracy:", accuracy)

# Print classification report
print("\n🔹 Classification Report:\n", classification_report(Y_test, Y_pred_nb))

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot(cmap=plt.cm.Purples)
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig("result/confusion_matrix_NB.png")
plt.show()

# Save the predictions to a CSV file
predictions_nb = pd.DataFrame(Y_pred_nb, columns=["Predicted"])
predictions_nb.to_csv("result/predctions_NB_model.csv", index=False)
