import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from itertools import product

# Download stock data using yfinance
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess data
data.reset_index(inplace=True)
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_normalized = scaler.fit_transform(closing_prices)

# Prepare dataset for LSTM
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length], self.data[idx+self.seq_length])

seq_length = 10  # Sequence length for LSTM input

dataset = StockDataset(closing_prices_normalized, seq_length)

# Hyperparameter grids
learning_rates = [0.0001, 0.001, 0.01]
hidden_sizes = [32, 64, 128]
num_layers_list = [1, 2, 3]
batch_sizes = [16, 32, 64]

def train_model(model, dataloader, epochs=20, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for seq, target in dataloader:
            seq, target = seq.to(device).float(), target.to(device).float()
            outputs = model(seq)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Define 1D CNN model
class CNN1DModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(32 * (seq_length - 4), output_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, seq_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models = [LSTMModel, GRUModel, CNN1DModel]
model_names = ['LSTM', 'GRU', 'CNN']

best_model = None
best_model_name = ""
best_mse = float('inf')
best_params = {}

# Hyperparameter tuning and cross-validation
for model_class, model_name in zip(models, model_names):
    for lr, hidden_size, num_layers, batch_size in product(learning_rates, hidden_sizes, num_layers_list, batch_sizes):
        if model_name == 'CNN':
            # CNN does not use hidden_size or num_layers
            model = model_class(input_size=1, output_size=1)
        else:
            model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
        
        fold = 1
        mse_scores = []
        print(f"Testing {model_name} with lr={lr}, hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}")
        for train_idx, val_idx in kfold.split(closing_prices_normalized):
            train_data = closing_prices_normalized[train_idx]
            val_data = closing_prices_normalized[val_idx]

            train_dataset = StockDataset(train_data, seq_length)
            val_dataset = StockDataset(val_data, seq_length)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Train model
            trained_model = train_model(model, train_loader, epochs=10, lr=lr)

            # Evaluate on validation set
            trained_model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(device).float(), target.to(device).float()
                    outputs = trained_model(seq)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            mse = mean_squared_error(val_targets, val_predictions)
            mse_scores.append(mse)
            print(f'{model_name} - Fold {fold}, MSE: {mse:.4f}')
            fold += 1

        avg_mse = np.mean(mse_scores)
        print(f'Average MSE for {model_name} with lr={lr}, hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}: {avg_mse:.4f}\n')

        # Track best model
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_model = model
            best_model_name = model_name
            best_params = {
                'learning_rate': lr,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'batch_size': batch_size
            }

print(f'Best Model: {best_model_name} with params {best_params} and MSE: {best_mse:.4f}')

# Train final best model and make predictions for visualization
best_model_instance = best_model.to(device)
final_dataset = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)
trained_best_model = train_model(best_model_instance, final_dataset, epochs=20, lr=best_params['learning_rate'])

# Make predictions with the best model
trained_best_model.eval()
with torch.no_grad():
    data_tensor = torch.tensor(closing_prices_normalized).float().to(device)
    best_model_predictions = []
    for i in range(len(data) - seq_length):
        seq = data_tensor[i:i+seq_length].unsqueeze(0)
        best_model_predictions.append(trained_best_model(seq).item())

# Inverse transform the normalized data
best_model_predicted_prices = scaler.inverse_transform(np.array(best_model_predictions).reshape(-1, 1))
actual_prices = closing_prices[seq_length:]

# Plot results for the best model
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Closing Prices')
plt.plot(best_model_predicted_prices, label=f'{best_model_name} Predicted Closing Prices', linestyle='--')
plt.title(f'{best_model_name} Predictions vs Actual Closing Prices')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
