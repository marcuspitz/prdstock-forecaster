import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Sample data
data = {
    'Product': [
        '2355107', '2358686', '2359456', '2359456', '2359456', '2359456', '2359456',
        '2359456', '2359456', '2359456', '2359456', '2359456', '2359456', '2359456',
        '2359456', '2359456', '2359456', '2359456', '2359456', '2359456', '2359456',
        '2359456'
    ],
    'Date': [
        '18/4/2023', '21/12/2023', '13/6/2024', '6/6/2024', '23/4/2024', '23/4/2024',
        '19/2/2024', '2/2/2024', '17/10/2023', '3/10/2023', '18/8/2023', '19/6/2023',
        '19/6/2023', '19/6/2023', '18/4/2023', '21/3/2023', '10/3/2023', '1/3/2023',
        '4/1/2023', '4/1/2023', '1/9/2022', '18/5/2022'
    ],
    'Quantity': [
        1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)

# Filter for a specific product
product_code = '2359456'
df_product = df[df['Product'] == product_code]

# Resample to monthly frequency, summing the quantities
df_product = df_product['Quantity'].resample('M').sum()

# Function to fit and forecast with SARIMA model
def fit_sarima(series):
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=12)
        return forecast
    except Exception as e:
        print(f"SARIMA Error: {e}")
        return None

# Function to fit and forecast with ARIMA model
def fit_arima(series):
    try:
        model = auto_arima(series, seasonal=False, trace=True)
        forecast = model.predict(n_periods=12)
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None

# Function to prepare data and train neural network
def train_neural_network(series):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

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
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[seq_length, 1]),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)

        forecast = []
        current_batch = X_test[0].reshape((1, seq_length, 1))
        for i in range(12):
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        forecast = scaler.inverse_transform(forecast)
        return forecast
    except Exception as e:
        print(f"Neural Network Error: {e}")
        return None

# Fit and forecast using the models
sarima_forecast = fit_sarima(df_product)
arima_forecast = fit_arima(df_product)
neural_network_forecast = train_neural_network(df_product)

# Check if all forecasts are generated successfully
if sarima_forecast is not None and arima_forecast is not None and neural_network_forecast is not None:
    plt.figure(figsize=(12, 8))

    plt.plot(df_product.index, df_product, label='Actual', marker='o')

    forecast_index = pd.date_range(start=df_product.index[-1], periods=12, freq='M')

    sarima_nn_forecast = sarima_forecast + neural_network_forecast.flatten()
    plt.plot(forecast_index, sarima_nn_forecast, label='SARIMA + Neural Network Forecast', linestyle='--')

    arima_nn_forecast = arima_forecast + neural_network_forecast.flatten()
    plt.plot(forecast_index, arima_nn_forecast, label='ARIMA + Neural Network Forecast', linestyle='-.')

    plt.title('Quantity Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("One or more forecasts were not generated successfully.")
