import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

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

# Main function to run the forecasting without regression-based combination
def main_forecasting(file_path, product_code):
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

    # Fit and forecast on the full data using the models
    sarima_forecast = fit_sarima(df_product)
    arima_forecast = fit_arima(df_product)
    neural_network_forecast = train_neural_network(df_product)

    # Combine the data into a DataFrame for exporting
    forecast_index = pd.date_range(start=df_product.index[-1], periods=12, freq='M')
    results_df = pd.DataFrame({
        'Actual': pd.concat([df_product, pd.Series([np.nan]*12, index=forecast_index)]),
        'SARIMA': pd.Series(sarima_forecast, index=forecast_index),
        'ARIMA': pd.Series(arima_forecast, index=forecast_index),
        'Neural Network': pd.Series(neural_network_forecast, index=forecast_index),
    })

    # Create an Excel file and add the data
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = 'Forecast Results'

    # Write the DataFrame to the worksheet
    for r in dataframe_to_rows(results_df, index=True, header=True):
        worksheet.append(r)

    # Create a LineChart
    chart = LineChart()
    chart.title = "Min safety stock"
    chart.style = 13
    chart.x_axis.title = "Date"
    chart.y_axis.title = "Quantity"

    # Add data to the chart
    data = Reference(worksheet, min_col=2, min_row=1, max_col=5, max_row=len(results_df) + 1)
    categories = Reference(worksheet, min_col=1, min_row=2, max_row=len(results_df) + 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    # Specify colors for the series
    colors = ['FF0000', '00FF00', '0000FF', 'FFFF00', 'FF00FF', '00FFFF']
    for i, series in enumerate(chart.series):
        series.graphicalProperties.line.solidFill = colors[i % len(colors)]

    # Add the chart to the worksheet
    worksheet.add_chart(chart, "A15")

    # Save the Excel file
    workbook.save("forecast_results.xlsx")

    print(f"Results and chart have been saved to forecast_results.xlsx")

# Example usage
file_path =  input("Enter the CSV file path: ") # 'C:\\Users\\marcus\\Downloads\\all-products-formatted.csv'  # Replace with the path to your CSV file
#file_path =  'C:\\Users\\marcus\\Downloads\\all-products-formatted-2.csv'
product_code = input("Enter the product code to filter: ")# '2359456'  # Replace with the desired product code
main_forecasting(file_path, product_code)
