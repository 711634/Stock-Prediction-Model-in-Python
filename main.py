import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'GS'

start = dt.datetime(2018, 1, 10)
train_end = dt.datetime(2022, 12, 31)
test_start = dt.datetime(2023, 1, 1)
end = dt.datetime.now()

data = yf.download(company, start=start, end=end)

# Separate data into training and test sets
train_data = data[data.index <= train_end]
test_data = data[data.index >= test_start]

# Prepare Training Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

prediction_days = 730

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_train_data)):
    x_train.append(scaled_train_data[x-prediction_days:x, 0])
    y_train.append(scaled_train_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build The Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Prepare Test Data
actual_prices = test_data['Close'].values
total_dataset = pd.concat((train_data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the Test Predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="blue", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# Predict Next 30 Days
last_prediction_data = model_inputs[-prediction_days:]

future_predictions = []

for _ in range(30):
    prediction_data = last_prediction_data.reshape((1, prediction_days, 1))
    next_day_prediction = model.predict(prediction_data)
    future_predictions.append(next_day_prediction[0][0])
    last_prediction_data = np.roll(last_prediction_data, -1)
    last_prediction_data[-1] = next_day_prediction[0][0]

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate dates for the next 30 days
last_date = test_data.index[-1]
next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# Plot the Predictions
plt.plot(next_dates, future_predictions, color="green", label="Next 30 Days Prediction")
plt.title(f"{company} Share Price Forecast for the Next 30 Days")
plt.xlabel('Date')
plt.ylabel(f'{company} Share Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
