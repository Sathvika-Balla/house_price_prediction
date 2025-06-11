import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("house_data.csv")

# Features and target
X = data[["Area (sqft)", "Bedrooms", "Age"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print predictions
print("Predictions:")
for i in range(len(X_test)):
    print(f"Predicted: ₹{round(y_pred[i])}, Actual: ₹{y_test.iloc[i]}")

# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
