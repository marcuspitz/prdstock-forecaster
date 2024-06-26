import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

# Function to set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

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
        return np.full(steps, np.nan)

# Function to fit and forecast with ARIMA model
def fit_arima(series, steps=12):
    try:
        model = auto_arima(series, seasonal=False, trace=True)
        forecast = model.predict(n_periods=steps)
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return np.full(steps, np.nan)

# Function to prepare data and train neural network
def train_neural_network(series, steps=12):
    try:
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                y = data[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        seq_length = 6
        X, y = create_sequences(series.values, seq_length)
        
        if len(X) < 2 or len(y) < 2:
            print("Not enough data to train the neural network.")
            return np.full(steps, np.nan)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        set_seed()  # Set random seed for reproducibility

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[seq_length, 1]),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)  # No activation function here to ensure output is not constrained
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)

        forecast = []
        current_batch = X_test[0].reshape((1, seq_length, 1))
        for i in range(steps):
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        return np.array(forecast).flatten()
    except Exception as e:
        print(f"Neural Network Error: {e}")
        return np.full(steps, np.nan)

# Function to handle NaNs
def handle_nans(df):
    return df.fillna(df.mean())

# Main function to run the forecasting with regression-based approach
def main_regression_based(file_path, product_code):
    df_product = read_and_preprocess_csv(file_path, product_code)

    # Ensure there is enough data
    if len(df_product) < 2:
        print("Not enough data for forecasting.")
        return

    # Split into training and validation sets
    train_size = int(len(df_product) * 0.8)
    train, validation = df_product[:train_size], df_product[train_size:]

    # Fit and forecast on the validation set using the models
    sarima_val_forecast = fit_sarima(train, steps=len(validation))
    arima_val_forecast = fit_arima(train, steps=len(validation))
    nn_val_forecast = train_neural_network(train, steps=len(validation))

    # Combine forecasts for the validation set
    combined_val_forecast_sarima_nn = pd.DataFrame({
        'SARIMA': sarima_val_forecast,
        'NN': nn_val_forecast
    }, index=validation.index)
    combined_val_forecast_arima_nn = pd.DataFrame({
        'ARIMA': arima_val_forecast,
        'NN': nn_val_forecast
    }, index=validation.index)

    # Handle NaNs in the combined forecasts
    combined_val_forecast_sarima_nn = handle_nans(combined_val_forecast_sarima_nn)
    combined_val_forecast_arima_nn = handle_nans(combined_val_forecast_arima_nn)

    # Train regression model on the validation set
    reg_sarima_nn = LinearRegression()
    reg_sarima_nn.fit(combined_val_forecast_sarima_nn, validation.values)
    
    reg_arima_nn = LinearRegression()
    reg_arima_nn.fit(combined_val_forecast_arima_nn, validation.values)

    # Fit and forecast on the full data using the models
    sarima_forecast = fit_sarima(df_product)
    arima_forecast = fit_arima(df_product)
    neural_network_forecast = train_neural_network(df_product)

    # Combine forecasts for the test set
    combined_forecast_sarima_nn = pd.DataFrame({
        'SARIMA': sarima_forecast,
        'NN': neural_network_forecast
    }, index=pd.date_range(start=df_product.index[-1], periods=12, freq='M'))
    combined_forecast_arima_nn = pd.DataFrame({
        'ARIMA': arima_forecast,
        'NN': neural_network_forecast
    }, index=pd.date_range(start=df_product.index[-1], periods=12, freq='M'))

    # Handle NaNs in the combined forecasts
    combined_forecast_sarima_nn = handle_nans(combined_forecast_sarima_nn)
    combined_forecast_arima_nn = handle_nans(combined_forecast_arima_nn)

    # Predict using the regression models
    reg_combined_forecast_sarima_nn = reg_sarima_nn.predict(combined_forecast_sarima_nn)
    reg_combined_forecast_arima_nn = reg_arima_nn.predict(combined_forecast_arima_nn)

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.plot(df_product.index, df_product, label='Actual', marker='o')

    forecast_index = pd.date_range(start=df_product.index[-1], periods=12, freq='M')

    # Plot individual models' forecasts
    plt.plot(forecast_index, sarima_forecast, label='SARIMA', linestyle='--', color='blue')
    plt.plot(forecast_index, arima_forecast, label='ARIMA', linestyle='-.', color='orange')
    plt.plot(forecast_index, neural_network_forecast, label='Neural Network', linestyle=':', color='green')

    # Plot combined forecasts using the regression model
    plt.plot(forecast_index, reg_combined_forecast_sarima_nn, label='SARIMA + Neural Network (Regression)', linestyle='-', color='red')
    plt.plot(forecast_index, reg_combined_forecast_arima_nn, label='ARIMA + Neural Network (Regression)', linestyle='-', color='purple')

    plt.title('Quantity Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)

    # Format x-axis to show mm/yyyy
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%Y'))

    plt.show()

# Example usage
file_path =  input("Enter the CSV file path: ") # 'C:\\Users\\marcus\\Downloads\\all-products-formatted.csv'  # Replace with the path to your CSV file
product_code = input("Enter the product code to filter: ")# '2359456'  # Replace with the desired product code
main_regression_based(file_path, product_code)
