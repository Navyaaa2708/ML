import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from data_preprocessing import split_data

# Load dataset

data = pd.read_csv("datasets/sample.csv")

X = data[["feature"]]
y = data["target"]

# Split data

X_train, X_test, y_train, y_test = split_data(X, y)

# Train model

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("R² Score:", score)

# Save model

with open("models/saved_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
