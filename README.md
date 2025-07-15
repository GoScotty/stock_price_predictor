import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load stock data (CSV should have 'Date' and 'Close' columns)
df = pd.read_csv('stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Use days as a feature
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
X = df[['Days']]
y = df['Close']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future prices
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='Actual Prices')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Predicted Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

# Predict next day's price
next_day = np.array([[df['Days'].max() + 1]])
next_price = model.predict(next_day)
print(f"Predicted price for next day: {next_price[0]:.2f}")

