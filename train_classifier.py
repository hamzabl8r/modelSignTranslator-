import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Safely load the data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# 2. Convert to numpy arrays
# Ensure all vectors are the same length (84 features for 2 hands)
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Total samples: {len(data)}")
print(f"Feature vector length: {data.shape[1]}") # Should be 84

# 3. Split into Training and Testing sets
# 'stratify' ensures each word has equal representation in train/test
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# 4. Initialize and Train the Model
# n_estimators=100 provides a good balance of speed and accuracy for words
model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
model.fit(x_train, y_train)

# 5. Evaluate the Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print("\n--- Training Results ---")
print(f'Overall Accuracy: {score * 100:.2f}%')

# Detailed report to see which words the model struggles with
print("\nClassification Report per Word:")
print(classification_report(y_test, y_predict))

report_dict = classification_report(y_test, y_predict, output_dict=True)
per_class = []

for label, values in report_dict.items():
    if label in ("accuracy", "macro avg", "weighted avg"):
        continue
    per_class.append({
        "label": label,
        "precision": float(values.get("precision", 0.0)),
        "recall": float(values.get("recall", 0.0)),
        "f1_score": float(values.get("f1-score", 0.0)),
        "support": int(values.get("support", 0)),
    })

per_class.sort(key=lambda item: item["f1_score"], reverse=True)

metrics_payload = {
    "accuracy": float(report_dict.get("accuracy", score)),
    "macro_avg": {
        "precision": float(report_dict.get("macro avg", {}).get("precision", 0.0)),
        "recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
        "f1_score": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
    },
    "weighted_avg": {
        "precision": float(report_dict.get("weighted avg", {}).get("precision", 0.0)),
        "recall": float(report_dict.get("weighted avg", {}).get("recall", 0.0)),
        "f1_score": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
    },
    "sample_count": int(len(data)),
    "test_sample_count": int(len(y_test)),
    "feature_count": int(data.shape[1]),
    "per_class": per_class,
}

with open('train_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_payload, f, ensure_ascii=True, indent=2)

print("Metrics saved as train_metrics.json")

# 6. Safely save the trained model
# We save the whole dict so we can add metadata later if needed
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")
