# prdstock-forecaster
This project forecasts product stock based on historical data.

## stacks
* ARIMA
* SARIMA
* Combined with Neural Networks
* Time series analysis

## getting started
* `pip install pandas numpy matplotlib statsmodels pmdarima tensorflow scikit-learn`
* `python main.py`

## build the executable
* First time only: `pip install pyinstaller`
* This is not necessary, since we have a specific spec: `pyinstaller --onefile script.py`
* `pyinstaller main-regression-stacking.spec`

## results and discussions
First impression is that regression, and stacking approaches are the final candidates.