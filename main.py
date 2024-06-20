import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Sample data
data = {
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='MS'),
    'Part Number': '00B46117PC1K0',
    'Description': 'FU 1000A 500V FWH Extra Quickblow',
    'Spare Parts': [12] * 12,
    'Crash Kit': [12] * 12,
    'Retrofit': [12] * 12
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Function to fit and forecast with SARIMA model
def fit_sarima(df, target_col):
    try:
        model = SARIMAX(df[target_col], order=(1, 1, 1), seasonal_order=(0, 0, 0, 12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=12)
        print(f"SARIMA forecast length: {len(forecast)}")
        return forecast
    except Exception as e:
        print(f"SARIMA Error: {e}")
        return None

# Function to fit and forecast with ARIMA model
def fit_arima(df, target_col):
    try:
        model = auto_arima(df[target_col], seasonal=False, trace=True)
        forecast = model.predict(n_periods=12)
        print(f"ARIMA forecast length: {len(forecast)}")
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None

# Function to prepare data and train neural network
def train_neural_network(df, target_col):
    try:
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[target_col]])

        # Create sequences for time series data
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data)-seq_length):
                x = data[i:i+seq_length]
                y = data[i+seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        seq_length = 6
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[seq_length, 1]),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)

        # Forecasting
        forecast = []
        current_batch = X_test[0].reshape((1, seq_length, 1))
        for i in range(12):  # Predict 12 steps ahead
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        forecast = scaler.inverse_transform(forecast)
        print(f"Neural Network forecast length: {len(forecast)}")
        return forecast
    except Exception as e:
        print(f"Neural Network Error: {e}")
        return None

# Example usage:
target_col = 'Spare Parts'

# SARIMA forecast
sarima_forecast = fit_sarima(df, target_col)
if sarima_forecast is None:
    print("Failed to generate SARIMA forecast.")

# ARIMA forecast
arima_forecast = fit_arima(df, target_col)
if arima_forecast is None:
    print("Failed to generate ARIMA forecast.")

# Neural Network forecast
neural_network_forecast = train_neural_network(df, target_col)
if neural_network_forecast is None:
    print("Failed to generate Neural Network forecast.")

# Check if all forecasts are generated successfully
if sarima_forecast is not None and arima_forecast is not None and neural_network_forecast is not None:
    # Plotting all forecasts together
    plt.figure(figsize=(12, 8))

    # Actual data
    plt.plot(df.index, df[target_col], label='Actual', marker='o')

    # Extend the index for forecasts
    forecast_index = pd.date_range(start=df.index[-1], periods=12, freq='MS')

    # SARIMA + Neural Network forecast
    sarima_nn_forecast = sarima_forecast + neural_network_forecast.flatten()
    plt.plot(forecast_index, sarima_nn_forecast, label='SARIMA + Neural Network Forecast', linestyle='--')

    # ARIMA + Neural Network forecast
    arima_nn_forecast = arima_forecast + neural_network_forecast.flatten()
    plt.plot(forecast_index, arima_nn_forecast, label='ARIMA + Neural Network Forecast', linestyle='-.')

    plt.title('Spare Parts Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Spare Parts')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("One or more forecasts were not generated successfully.")
