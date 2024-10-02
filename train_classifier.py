import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle','rb'))

# Determine the fixed length (e.g., 42 for 21 landmarks with x and y coordinates)
fixed_length = 42

# Pad or truncate sequences to the fixed length
data = []
for item in data_dict['data']:
    if len(item) < fixed_length:
        item.extend([0] * (fixed_length - len(item)))  # Pad with zeros
    elif len(item) > fixed_length:
        item = item[:fixed_length]  # Truncate to fixed length
    data.append(item)

# Convert to NumPy array
data = np.asarray(data)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"Accuracy is {score * 100}%")

f = open('model.p', 'wb')
pickle.dump({'model':model}, f)
f.close()
