import yfinance as yf  # Importing the yfinance library to fetch stock data
import matplotlib.pyplot as plt  # Importing matplotlib for plotting data

# Fetch historical stock data for a company (e.g., Apple Inc. with ticker symbol 'AAPL')
ticker = 'AAPL'  # Define the ticker symbol for Apple Inc.
start_date = '2020-01-01'  # Define the start date for data collection
end_date = '2024-01-01'  # Define the end date for data collection

# Download stock data using the yfinance library
data = yf.download(ticker, start=start_date, end=end_date)

# Reset index to flatten the DataFrame structure
data.reset_index(inplace=True)  # Flatten the DataFrame for easier access to columns

# Display the first few rows of the data to understand its structure
print("Raw Data:")
print(data.head())  # Print the first five rows of the DataFrame

# Check for missing values in the dataset
print("\nMissing Values:")
print(data.isnull().sum())  # Print the count of missing values for each column

# Plot the closing prices over time to visualize stock performance
plt.figure(figsize=(14, 7))  # Set the figure size for the plot
plt.plot(data['Close'], label='Closing Price', color='blue')  # Plot closing prices in blue
plt.title(f'{ticker} Closing Prices from {start_date} to {end_date}')  # Add a title to the plot
plt.xlabel('Date')  # Label the x-axis
plt.ylabel('Price ($)')  # Label the y-axis
plt.legend()  # Add a legend to the plot
plt.grid()  # Enable grid lines for better readability
plt.show()  # Display the plot

# Display summary statistics to understand the distribution of the data
print("\nSummary Statistics:")
print(data.describe())  # Print summary statistics like count, mean, min, and max values

# Calculate and visualize moving averages to identify trends in the data
data['5-day MA'] = data['Close'].rolling(window=5).mean()  # Calculate 5-day moving average
data['20-day MA'] = data['Close'].rolling(window=20).mean()  # Calculate 20-day moving average

# Create a plot for closing prices and moving averages
plt.figure(figsize=(14, 7))  # Set the figure size for the plot
plt.plot(data['Close'], label='Closing Price', color='blue')  # Plot closing prices in blue
plt.plot(data['5-day MA'], label='5-Day Moving Average', color='red')  # Plot 5-day MA in red
plt.plot(data['20-day MA'], label='20-Day Moving Average', color='green')  # Plot 20-day MA in green
plt.title(f'{ticker} Closing Prices and Moving Averages')  # Add a title to the plot
plt.xlabel('Date')  # Label the x-axis
plt.ylabel('Price ($)')  # Label the y-axis
plt.legend()  # Add a legend to the plot
plt.grid()  # Enable grid lines for better readability
plt.show()  # Display the plot
