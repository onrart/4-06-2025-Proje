import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import predict_house_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    model_path = "models/house_price_model.pkl"
    scaler_path = "models/feature_scaler.pkl"
    features_path = "models/feature_names.pkl"

    if not (
        os.path.exists(model_path)
        and os.path.exists(scaler_path)
        and os.path.exists(features_path)
    ):
        st.error(
            "Model files not found. Please run save_model.py first to create the model files."
        )
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


def main():

    st.title("🏠 House Price Prediction")
    st.markdown(
        """
    This app predicts house prices based on features lise square footage, number of bedrooms, etc.
    Enter the details of house below to get a price estimate.
    """
    )

    with st.spinner("Loading model..."):
        model, scaler, feature_names = load_model()

    st.sidebar.header("House Features")

    sqft = st.sidebar.slider("Square Footage", 500, 5000, 2000, 100)
    bedrooms = st.sidebar.slider("Number of bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Number of bathrooms", 1.0, 7.0, 2.0, 0.5)
    age = st.sidebar.slider("Age of House (years)", 0, 100, 15)
    lot_size = st.sidebar.slider("Lot Size (sq ft)", 1000, 20000, 8000, 500)

    features = {
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "lot_size": lot_size,
    }
    if st.sidebar.button("Predict Price"):
        predicted_price = predict_house_price(model, scaler, features)

        st.header("Prediction Results")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.metric(label="Estimated Hosue Price", value=f"{predicted_price:.2f}")

            st.subheader("Feature Importance")
            importance = pd.DataFrame(
                {"Feature": feature_names, "Coefficient": model.coef_}
            )

            importance["Absolute"] = abs(importance["Coefficient"])
            importance = importance.sort_values(by="Absolute", ascending=False)

            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            bars = ax.barh(importance["Feature"], importance["Absolute"])
            ax.set_xlabel("Absolute Coefficient Value")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        with col2:

            st.subheader("House Features")

            features_df = pd.DataFrame(features, index=[0])
            st.table(features_df)

            st.subheader("House Visualization")
            fig, ax = plt.figure(figsize=(8, 6)), plt.axes()

            house_width = np.sqrt(sqft / 2)
            house_height = np.sqrt(sqft / 2)

            rect = plt.Rectangle(
                (0, 0),
                house_width,
                house_height,
                facecolor="lightblue",
                edgecolor="blue",
                alpha=0.7,
            )
            ax.add_patch(rect)

            plt.plot(
                [0, house_width / 2, house_width],
                [house_height, house_height + house_height / 2, house_height],
                "r-",
            )

            window_size = house_width / 10

            for i in range(min(bedrooms, 5)):

                x = house_width * (i + 1) / (bedrooms + 1) - window_size / 2
                y = house_height / 2
                window = plt.Rectangle(
                    (x, y),
                    window_size,
                    window_size,
                    facecolor="white",
                    edgecolor="black",
                )
                ax.add_patch(window)

            # Draw door
            door_width = house_width / 8
            door_height = house_height / 3
            door = plt.Rectangle(
                (house_width / 2 - door_width / 2, 0),
                door_width,
                door_height,
                facecolor="brown",
                edgecolor="black",
            )
            ax.add_patch(door)

            # Set plot limits and remove axes
            ax.set_xlim(-house_width / 5, house_width + house_width / 5)
            ax.set_ylim(
                -house_height / 10, house_height + house_height / 2 + house_height / 10
            )
            ax.set_aspect("equal")
            ax.axis("off")

            st.pyplot(fig)

        # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
    This app uses a Linear Regression model to predict house prices.
    
    The model considers:
    - Square footage
    - Number of bedrooms
    - Number of bathrooms
    - Age of the house
    - Lot size
    """
    )


if __name__ == "__main__":
    main()
