import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import KNNImputer

# Load Data
df = pd.read_csv("World Energy.csv")


def forecast_energy(country, year):
    metric = "electricity_demand"

    # Filter data for the selected country
    data = df[df["country"] == country][["year", metric]].dropna()
    data = data.sort_values(by="year")
    data["year"] = data["year"].astype(int)  # Ensure year is integer
    data.set_index("year", inplace=True)
    data.index = pd.RangeIndex(start=data.index.min(), stop=data.index.max() + 1, step=1)

    # Handle missing values using KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    data[metric] = imputer.fit_transform(data[[metric]])
    data[metric] = data[metric].astype(float)

    # Ensure forecast year is in the future
    last_year = data.index[-1]
    if year <= last_year:
        raise ValueError(f"Year must be greater than {last_year}. Please enter a future year.")

    # Train ARIMA Model
    model = ARIMA(data[metric], order=(3, 1, 3))  # (p,d,q) values can be tuned
    model_fit = model.fit()

    # Forecast Future Values
    future_years = np.arange(last_year + 1, year + 1)
    forecast = model_fit.forecast(steps=len(future_years))

    return data, future_years, forecast


# Streamlit UI
st.title("Energy Consumption Forecast")

country = st.text_input("Enter Country:")
year = st.number_input("Enter Year:", min_value=2000, step=1)

if st.button("Predict"):
    if country and year:
        try:
            data, future_years, forecast = forecast_energy(country, int(year))
            st.success(f"Predicted electricity demand in {year}: {forecast[-1]:.2f}")

            # Plot Historical Data and Forecast
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data["electricity_demand"], marker='o', linestyle='-', label="Historical Data")
            ax.plot(future_years, forecast, marker='o', linestyle='dashed', label="Forecast")
            ax.set_xlabel("Year")
            ax.set_ylabel("Electricity Demand")
            ax.set_title(f"Electricity Demand Forecast for {country}")
            ax.legend()
            ax.grid()

            st.pyplot(fig)
        except ValueError as e:
            st.error(str(e))
