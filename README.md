# SkillCraft-Task1
# House Price Prediction using Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Features and Target
X = data.select_dtypes(include=["int64", "float64"]).drop("SalePrice", axis=1)
y = data["SalePrice"]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("\nHouse Price Prediction Completed Successfully!")
print("Mean Absolute Error:", mae)
