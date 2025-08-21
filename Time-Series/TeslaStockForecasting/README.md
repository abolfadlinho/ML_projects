# Tesla Stock Forecasting

This project demonstrates time series forecasting of Tesla's stock price using an LSTM neural network. The workflow includes data loading, exploration, preprocessing, model training, evaluation, prediction, and future forecasting.

## Workflow Overview

1. **Data Loading & Exploration**

   - Loads Tesla stock data from CSV.
   - Explores data with `head()`, `info()`, `describe()`, and visualizes Open, Close, Volume, and daily returns.

2. **Preprocessing**

   - Selects the `Close` price for prediction.
   - Normalizes data using MinMaxScaler.
   - Splits data into training (75%) and testing (25%) sets.
   - Creates time series sequences (60 time-steps) for LSTM input.

3. **Model Building & Training**

   - Builds an LSTM model with stacked LSTM and Dense layers.
   - Compiles with Adam optimizer and MSE loss.
   - Trains with early stopping callback.

4. **Evaluation & Prediction**

   - Plots training loss and mean absolute error.
   - Prepares test sequences and predicts closing prices.
   - Inverse transforms predictions and calculates RMSE.
   - Visualizes actual vs. predicted prices.

5. **Forecasting Next 30 Days**
   - Uses the trained model to forecast the next 30 days of closing prices.
   - Creates a forecasted DataFrame and visualizes future predictions.

## Key Code Snippet: LSTM Model Structure

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(64, return_sequences=False),
    Dense(32),
    Dense(16),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics="mean_absolute_error")
```

## Requirements

- Python
- pandas
- numpy
- seaborn
- matplotlib
- keras
- scikit-learn

## Usage

1. Install dependencies:
   ```powershell
   pip install pandas numpy seaborn matplotlib keras scikit-learn
   ```
2. Run the notebook `tesla-stock-forecasting.ipynb` for step-by-step execution.

## References

- [Tesla Stock Data](https://www.kaggle.com/datasets)
- [Keras Documentation](https://keras.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

For details, see the notebook: `tesla-stock-forecasting.ipynb`.
