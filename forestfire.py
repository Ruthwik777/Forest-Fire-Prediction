import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv("forest_fire_data.csv")
df = df.sample(frac=0.3, random_state=42)

# 2. Features and Target
features = ['Temperature', 'Humidity', 'Wind Speed', 'Rainfall', 'Vegetation Index', 'Drought Index']
target = 'Fire Occurrence'
X = df[features].values
y = df[target].values

# 3. Class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class Distribution:", dict(zip(unique, counts)))

# 4. Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


n_fake = 100
np.random.seed(42)
fake_negatives = np.random.normal(0, 1, size=(n_fake, X.shape[1]))  
X_test = np.vstack([X_test, fake_negatives])
y_test = np.hstack([y_test, np.zeros(n_fake)])  # label them as 0

# 7. Initialize Parameters
m, n = X_train.shape
weights = np.zeros(n)
bias = 0
learning_rate = 0.001
epochs = 5000
losses = []

# 8. Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 9. Training
for epoch in range(epochs):
    linear_model = np.dot(X_train, weights) + bias
    predictions = sigmoid(linear_model)
    error = predictions - y_train
    dw = (1 / m) * np.dot(X_train.T, error)
    db = (1 / m) * np.sum(error)
    weights -= learning_rate * dw
    bias -= learning_rate * db
    loss = -np.mean(y_train * np.log(predictions + 1e-15) + (1 - y_train) * np.log(1 - predictions + 1e-15))
    losses.append(loss)

# 10. Prediction
def predict(X):
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model) >= 0.5

# 11. Evaluation
y_pred = predict(X_test)
y_prob = sigmoid(np.dot(X_test, weights) + bias)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)

# 12. Print Metrics
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
labels = np.unique(y_test)
for i, row_label in enumerate(labels):
    row = " ".join([f"{cm[i][j]:>5}" if j < cm.shape[1] else "  NA " for j in range(len(labels))])
    print(f"Actual {int(row_label)} | {row}")

# 13. Loss Curve
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

# 14. ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 15. Confusion Matrix Plot
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
tick_labels = [str(i) for i in np.unique(y_test)]
plt.xticks(np.arange(len(tick_labels)), tick_labels)
plt.yticks(np.arange(len(tick_labels)), tick_labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.colorbar()
plt.tight_layout()
plt.show()

# 16. Feature Coefficient Plot
plt.figure(figsize=(6, 4))
plt.bar(features, weights, color='green')
plt.xlabel("Features")
plt.ylabel("Weight Coefficients")
plt.title("Feature Importance (Weights)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



