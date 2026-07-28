Train the LSTM load predictor on sharding metrics time series.
Multi‑task: regression (next metrics) + classification (overload).
Exports quantized .pt.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import os
import argparse

class ShardingDataset(Dataset):
    def __init__(self, h5_path):
        self.file = h5py.File(h5_path, 'r')
        self.samples = []
        for shard in self.file.values():
            seqs = shard['sequences']
            targets = shard['targets']
            labels = shard['overload_labels']
            for i in range(len(seqs)):
                self.samples.append((seqs[i], targets[i], labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target, label = self.samples[idx]
        seq = torch.from_numpy(seq).float()
        target = torch.from_numpy(target).float()
        label = torch.tensor(label).float().unsqueeze(0)
        return seq, target, label

class ShardLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_metrics = nn.Linear(hidden_dim, input_dim)
        self.fc_overload = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        metrics = self.fc_metrics(out)
        overload = torch.sigmoid(self.fc_overload(out))
        return metrics, overload

def train_lstm(data_path, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShardLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()

    dataset = ShardingDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for seq, target, label in tqdm(dataloader):
            seq, target, label = seq.to(device), target.to(device), label.to(device)
            pred_metrics, pred_overload = model(seq)
            loss_reg = criterion_reg(pred_metrics, target)
            loss_cls = criterion_cls(pred_overload, label)
            loss = loss_reg + loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg loss {total_loss/len(dataloader):.4f}")

    # Quantize
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), "lstm_1.1mb_quantized.pt")
    print("Saved LSTM, size: {:.2f} MB".format(
        os.path.getsize("lstm_1.1mb_quantized.pt") / 1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to sharding_*.h5")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_lstm(args.data, args.epochs)
