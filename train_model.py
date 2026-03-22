import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    os.makedirs("models", exist_ok=True)

# 1. Load dataset
    file_path = "data/housing.csv"
    df = pd.read_csv(file_path)


    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

# 2. Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

# 3. Drop missing rows for beginner simplicity
    df = df.dropna()

# 4. Separate features and target
    target_column = "price"
    X = df.drop(columns=[target_column])
    y = df[target_column]

# 5. Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

# Save feature column names for Streamlit app
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "models/feature_columns.pkl")

# 6. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
)

# 7. Create model


    model = LinearRegression()

# 8. Train model
    model.fit(X_train, y_train)

# 9. Predict
    y_pred = model.predict(X_test)

# 10. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2 : {r2:.4f}")

# 11. Save model
    joblib.dump(model, "models/linear_model.pkl")
    print("\nModel saved as models/linear_model.pkl")
    print("Feature columns saved as models/feature_columns.pkl")

if __name__ == "__main__":
    main()  
