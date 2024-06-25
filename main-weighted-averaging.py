import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Function to read CSV file and preprocess data
def read_and_preprocess_csv(file_path, product_code):
    df = pd.read_csv(file_path, delimiter=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    
    # Filter for a specific product
    df_product = df[df['Product'] == product_code]
    
    # Resample to monthly frequency, summing the quantities
    df_product = df_product['Quantity'].resample('M').sum()
    return df_product

# Function to fit and forecast with SARIMA model
def fit_sarima(series, steps=12):
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"SARIMA Error: {e}")
        return None

# Function to fit and forecast with ARIMA model
def fit_arima(series, steps=12):
    try:
        model = auto_arima(series, seasonal=False, trace=True)
        forecast = model.predict(n_periods=steps)
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None

# Function to prepare data and train neural network
def train_neural_network(series, steps=12):
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
        for i in range(steps):
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        forecast = scaler.inverse_transform(forecast)
        return forecast.flatten()
    except Exception as e:
        print(f"Neural Network Error: {e}")
        return None

# Function to calculate weights based on validation performance
def calculate_weights(val_errors):
    total_error = sum(val_errors)
    weights = [(total_error - e) / total_error for e in val_errors]
    return weights

# Main function to run the forecasting
def main(file_path, product_code):
    df_product = read_and_preprocess_csv(file_path, product_code)

    # Split into training and validation sets
    train_size = int(len(df_product) * 0.8)
    train, validation = df_product[:train_size], df_product[train_size:]

    # Fit and forecast on the validation set using the models
    sarima_val_forecast = fit_sarima(train, steps=len(validation))
    arima_val_forecast = fit_arima(train, steps=len(validation))
    nn_val_forecast = train_neural_network(train, steps=len(validation))

    # Calculate validation errors
    val_errors = [
        mean_squared_error(validation, sarima_val_forecast),
        mean_squared_error(validation, arima_val_forecast),
        mean_squared_error(validation, nn_val_forecast)
    ]

    # Calculate weights based on validation errors
    weights = calculate_weights(val_errors)

    # Fit and forecast on the full data using the models
    sarima_forecast = fit_sarima(df_product)
    arima_forecast = fit_arima(df_product)
    neural_network_forecast = train_neural_network(df_product)

    # Check if all forecasts are generated successfully
    if sarima_forecast is not None and arima_forecast is not None and neural_network_forecast is not None:
        plt.figure(figsize=(12, 8))

        plt.plot(df_product.index, df_product, label='Actual', marker='o')

        forecast_index = pd.date_range(start=df_product.index[-1], periods=12, freq='M')

        # Combine SARIMA and Neural Network forecasts using weighted averaging
        sarima_nn_combined = weights[0] * sarima_forecast + weights[2] * neural_network_forecast
        plt.plot(forecast_index, sarima_nn_combined, label='SARIMA + Neural Network', linestyle='--')

        # Combine ARIMA and Neural Network forecasts using weighted averaging
        arima_nn_combined = weights[1] * arima_forecast + weights[2] * neural_network_forecast
        plt.plot(forecast_index, arima_nn_combined, label='ARIMA + Neural Network', linestyle='-.')

        plt.title('Quantity Forecast Comparison')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)

        # Format x-axis to show mm/yyyy
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%Y'))

        plt.show()
    else:
        print("One or more forecasts were not generated successfully.")

# Example usage
file_path = 'C:\\Users\\marcus\\Downloads\\all-products-formatted.csv'  # Replace with the path to your CSV file
product_code = '2359456'  # Replace with the desired product code
main(file_path, product_code)
