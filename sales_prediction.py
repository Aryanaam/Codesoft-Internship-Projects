import pandas as pd
import numpy as np
import argparse
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def main(data_path, target_column):

    print(f"🔹 Loading dataset from: {data_path}")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

  
    print("\n✅ Dataset loaded successfully")
    print("Shape of dataset:", df.shape)

    print("\n📊 Full Dataset:")
    print(df.to_string())


    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n✅ Model Training Complete")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

 
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/sales_model.joblib")
    print("\n💾 Model saved to: outputs/sales_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV/XLSX file")
    parser.add_argument("--target", type=str, required=True, help="Target column name (e.g., Sales)")
    args = parser.parse_args()

    main(args.data, args.target)
