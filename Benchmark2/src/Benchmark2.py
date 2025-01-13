import yfinance as yf  # For downloading stock data
from sklearn.model_selection import train_test_split    # To split data into training and testing sets
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt  # Importing plotting library for visualization

ticker = 'AAPL'  # Setting the stock ticker for Apple Inc.
start_date = '2020-01-01'  # Defining the start date for data collection
end_date = '2024-01-01'  # Defining the end date for data collection

# Download stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Reset index and create the next day's closing price column
data.reset_index(inplace=True)  # Reset index to flatten the DataFrame structure
data['Next Close'] = data['Close'].shift(-1)    # Shift the 'Close' prices up by one row
data.dropna(inplace=True)   # Dropping the last row which will have NaN in 'Next Close' due to the shift

# Features (X) and Labels (y) setup
X = data[['Close']]  # Using the current closing price as the feature
y = data['Next Close']  # The target variable is the next closing price

# Splitting the dataset into training and testing sets with an 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()  # Instantiating the Linear Regression model
lr_model.fit(X_train, y_train)  # Training the model on the training set
lr_predictions = lr_model.predict(X_test)  # Making predictions on the test set

# Calculating performance metrics for Linear Regression
lr_mse = mean_squared_error(y_test, lr_predictions)  # Mean Squared Error
lr_r2 = r2_score(y_test, lr_predictions)  # R-squared score

# Decision Tree Regressor
dt_model = DecisionTreeRegressor()  # Instantiating the Decision Tree model
dt_model.fit(X_train, y_train)  # Training the model
dt_predictions = dt_model.predict(X_test)  # Making predictions

# Calculating performance metrics for Decision Tree
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

# Random Forest Regressor
rf_model = RandomForestRegressor()  # Instantiating the Random Forest model
rf_model.fit(X_train, y_train)  # Training the model
rf_predictions = rf_model.predict(X_test)  # Making predictions

# Calculating performance metrics for Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Printing the performance comparison for each model
print("\nModel Performance Comparison:")
print(f"Linear Regression - MSE: {lr_mse:.2f}, R^2: {lr_r2:.2f}")  # Displaying metrics for Linear Regression
print(f"Decision Tree - MSE: {dt_mse:.2f}, R^2: {dt_r2:.2f}")  # Displaying metrics for Decision Tree
print(f"Random Forest - MSE: {rf_mse:.2f}, R^2: {rf_r2:.2f}")  # Displaying metrics for Random Forest

# Linear Regression
plt.figure(figsize=(14, 7))
plt.scatter(y_test, lr_predictions, color='blue', label='Predicted vs Actual (Linear Regression)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Linear Regression Predictions vs Actual Closing Prices')
plt.xlabel('Actual Closing Prices')
plt.ylabel('Predicted Closing Prices')
plt.legend()
plt.grid()
plt.show()

# Decision Tree
plt.figure(figsize=(14, 7))
plt.scatter(y_test, dt_predictions, color='orange', label='Predicted vs Actual (Decision Tree)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Decision Tree Predictions vs Actual Closing Prices')
plt.xlabel('Actual Closing Prices')
plt.ylabel('Predicted Closing Prices')
plt.legend()
plt.grid()
plt.show()

# Random Forest
plt.figure(figsize=(14, 7))
plt.scatter(y_test, rf_predictions, color='purple', label='Predicted vs Actual (Random Forest)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Random Forest Predictions vs Actual Closing Prices')
plt.xlabel('Actual Closing Prices')
plt.ylabel('Predicted Closing Prices')
plt.legend()
plt.grid()
plt.show()
