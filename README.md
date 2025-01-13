# Stock Price Prediction

## Overview
This project aims to predict the closing price of Apple Inc. (AAPL) stock using both traditional machine learning techniques and advanced deep learning models. The goal was to evaluate and compare models like Linear Regression, Decision Tree, Random Forest, LSTM, GRU, and CNN to determine the best approach for stock price prediction.

## Table of Contents
1. [Project Description](#project-description)
2. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
3. [Machine Learning Models](#machine-learning-models)
4. [Results](#results)
5. [Installation Instructions](#installation-instructions)

## Project Description
The project focuses on predicting the next day's closing price of Apple stock (AAPL). The dataset spans from January 1, 2020, to January 1, 2024, and includes daily stock prices retrieved using the `yfinance` library. The primary goal is to compare different machine learning models to assess their prediction accuracy based on Mean Squared Error (MSE) and R-squared values.

## Data Collection and Preprocessing
- **Data Collection**: Stock data for Apple (AAPL) was collected using the `yfinance` library for the period between January 1, 2020, and January 1, 2024.
- **Data Cleaning**: Missing values were handled using the `.isnull().sum()` function. No significant missing data was found.
- **Data Visualization**: 
  - Closing prices were visualized to observe trends over time.
  - Moving averages (5-day and 20-day) were calculated and plotted to highlight trends.
- **Feature Engineering**: The dataset uses the current day's closing price as the feature, and the target variable is the next day's closing price.

## Machine Learning Models
The project utilizes both traditional and advanced machine learning models:

### Traditional Models:
1. **Linear Regression**
2. **Decision Tree**
3. **Random Forest**

### Advanced Models:
1. **LSTM (Long Short-Term Memory)**
2. **GRU (Gated Recurrent Unit)**
3. **CNN (Convolutional Neural Network)**

Each model was trained and evaluated on the Apple stock dataset. The models were tested using 5-fold cross-validation to assess their performance, and various hyperparameters were tuned for the advanced models to optimize results.

## Results
- **Benchmark 1 (Data Preprocessing and Exploratory Analysis)**: The dataset was cleaned and visualized, with moving averages showing trends in the data.
- **Benchmark 2 (Traditional Models)**:
  - **Linear Regression**: MSE = 6.97, R² = 0.99
  - **Decision Tree**: MSE = 15.81, R² = 0.99
  - **Random Forest**: MSE = 12.61, R² = 0.99
  Linear Regression outperformed the other models in terms of accuracy.
- **Benchmark 3 (Advanced Models before Hyperparameter Tuning)**:
  - **LSTM**: MSE = 0.0028, R² = 0.9382
  - **GRU**: MSE = 0.0021, R² = 0.9542 (Best performance)
  - **CNN**: MSE = 0.0026, R² = 0.9431
  GRU performed best, with the lowest MSE and highest R².
- **Benchmark 3 (Advanced Models after Hyperparameter Tuning)**:
  - The best model after hyperparameter tuning was **LSTM** with the following parameters:
    - Learning Rate: 0.01
    - Hidden Size: 128
    - Number of Layers: 3
    - Batch Size: 16
  - This model achieved the lowest MSE of 0.0023, making it the most accurate model.

## Installation Instructions
To run this project, you will need the following Python libraries:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `tensorflow`
- `keras`

To install the required libraries, run the following command:
```bash
pip install -r requirements.txt
