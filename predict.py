import pandas as pd
import joblib
import argparse

def main(model_path, input_path):
    # Load model
    model = joblib.load(model_path)
    print(f"🔹 Loaded model from {model_path}")

    # Load new data
    df_new = pd.read_csv(input_path)
    print("\n📊 New Data Preview:")
    print(df_new.head())

    # Predict
    predictions = model.predict(df_new)
    df_new['Predicted_Sales'] = predictions

    # Save results
    output_path = "outputs/predictions.csv"
    df_new.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--input", type=str, required=True, help="Path to new input CSV file")
    args = parser.parse_args()

    main(args.model, args.input)
