import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load predictions and ground truth
predictions = np.loadtxt("predictions.txt", dtype=int)
ground_truth = np.loadtxt("/path/to/ground_truth.txt", dtype=int)  # Replace with ground truth file path

# Ensure both have the same length
assert len(predictions) == len(ground_truth), "Mismatch between predictions and ground truth length"

# Calculate evaluation metrics
accuracy = accuracy_score(ground_truth, predictions)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(ground_truth, predictions))

print("Confusion Matrix:")
print(confusion_matrix(ground_truth, predictions))
