import pickle
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

# 6. Safely save the trained model
# We save the whole dict so we can add metadata later if needed
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")