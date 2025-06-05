import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """
    Load housing data from a CSV file
    If you dont have a dataset this func provieds a sample synthetic dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded from {filepath}")

    except:
        print("Creating synthetic data for demonstration")
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000

        # Features : squre footage, bedrooms, bathrooms, age of house, lot size

        sqft = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples) + np.random.random(n_samples)
        age = np.random.randint(0, 50, n_samples)
        lot_size = np.random.normal(8000, 2000, n_samples)

        # Target: house price
        price = (
            100000
            + 150 * sqft
            + 15000 * bedrooms
            + 25000 * bathrooms
            - 1000 * age
            + 2 * lot_size
        )
        price = np.random.normal(price, 50000, n_samples)  # add noise

        # Create DataFrame
        data = pd.DataFrame(
            {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age": age,
                "lot_size": lot_size,
                "price": price,
            }
        )

    return data


def preprocess_data(data):

    data = data.dropna()

    X = data.drop("price", axis=1)
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaler features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def train_model(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, feature_names):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(
        f"Model Performance:\n Mean Squared Error:{mse:.2f}\n Root Mean Squared Error:{rmse:.2f}\n R² Score :{r2:.4f}"
    )

    print("\nFeature Coefficients:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef:.2f}")

    print(f"Intercept: {model.intercept_:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")
    plt.tight_layout()
    plt.savefig("prediction_results.png")
    print("Plot saved as  'prediction_results.png'")

    return rmse, r2


def predict_house_price(model, scaler, features):

    features_df = pd.DataFrame([features])

    features_scaled = scaler.transform(features_df)

    predicted_price = model.predict(features_scaled)[0]

    return predicted_price


def main():
    data = load_data("house_data.csv")

    print("\nData Summary")

    print(data.describe())

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, feature_names)

    example_house = {
        "sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 10,
        "lot_size": 9000,
    }

    predicted_price = predict_house_price(model, scaler, example_house)

    print(f"\nExample Prediction:")
    print(f"House features: {example_house}")
    print(f"Predicted Price: ${predicted_price:.2f}")


if __name__ == "__main__":
    main()
