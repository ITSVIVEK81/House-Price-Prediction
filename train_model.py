import pandas as pd

# Load Dataset
data = pd.read_csv("dataset/house_price.csv")

# Show first 5 rows
print("First 5 Rows:")
print(data.head())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Convert categorical columns into numeric values
data = pd.get_dummies(data, drop_first=True)

print("\nAfter Encoding:")
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Features and Target
X = data.drop("price", axis=1)
y = data["price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# R² Scores
print("\nLinear Regression R² Score:", r2_score(y_test, lr_pred))
print("Decision Tree R² Score:", r2_score(y_test, dt_pred))
print("Random Forest R² Score:", r2_score(y_test, rf_pred))
import joblib

# Save the best model
joblib.dump(rf, "model/model.pkl")

print("Model saved successfully!")
print(data.columns)
