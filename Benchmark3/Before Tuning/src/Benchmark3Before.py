import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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

# Prepare dataset for LSTM, GRU, and CNN
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length], self.data[idx+self.seq_length])

seq_length = 10  # Sequence length for input
dataset = StockDataset(closing_prices_normalized, seq_length)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
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

# Train model
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

# Evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for seq, target in dataloader:
            seq, target = seq.to(device).float(), target.to(device).float()
            outputs = model(seq)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.cpu().numpy())
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, r2, predictions, targets

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models = {'LSTM': LSTMModel(), 'GRU': GRUModel(), 'CNN': CNN1DModel()}
results = {}
all_fold_results = {}

for model_name, model in models.items():
    print(f'Training {model_name} model with cross-validation...')
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(closing_prices_normalized)):
        train_data = closing_prices_normalized[train_idx]
        val_data = closing_prices_normalized[val_idx]

        train_dataset = StockDataset(train_data, seq_length)
        val_dataset = StockDataset(val_data, seq_length)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        trained_model = train_model(model, train_loader)
        mse, r2, predictions, targets = evaluate_model(trained_model, val_loader)
        fold_results.append({'mse': mse, 'r2': r2, 'predictions': predictions, 'targets': targets})
        print(f'Fold {fold + 1} - MSE: {mse:.4f}, R^2: {r2:.4f}')
    
    # Average results over folds
    avg_mse = np.mean([result['mse'] for result in fold_results])
    avg_r2 = np.mean([result['r2'] for result in fold_results])
    results[model_name] = {'mse': avg_mse, 'r2': avg_r2}
    all_fold_results[model_name] = fold_results
    print(f'{model_name} - Average MSE: {avg_mse:.4f}, Average R^2: {avg_r2:.4f}\n')

# Plot results
for model_name, fold_results in all_fold_results.items():
    for fold, fold_result in enumerate(fold_results):
        predicted_prices = scaler.inverse_transform(np.array(fold_result['predictions']).reshape(-1, 1))
        actual_prices = scaler.inverse_transform(np.array(fold_result['targets']).reshape(-1, 1))
        plt.figure(figsize=(14, 7))
        plt.plot(actual_prices, label='Actual Closing Prices')
        plt.plot(predicted_prices, label=f'{model_name} Fold {fold + 1} Predicted Closing Prices', linestyle='--')
        plt.title(f'{model_name} Fold {fold + 1} Predictions vs Actual Closing Prices')
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        plt.show()
