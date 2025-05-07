import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import random

class PerFileLSTMDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, "r") as f:
            for line in f:
                if "||" not in line:
                    continue
                x_str, y_str = line.strip().split("||")
                x = np.fromstring(x_str, sep=",")
                y = np.fromstring(y_str, sep=",").sum()
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor([y], dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dims=[256, 128, 64], dropout=0.05):
        super(DNN, self).__init__()
        layers = []
        last_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(last_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hdim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1)
        return self.net(x)

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def train_and_validate_on_split_file(model, filepath, optimizer, criterion, device, batch_size):
    dataset = PerFileLSTMDataset(filepath)
    if len(dataset) < 10:
        return 0.0, 0.0, []

    train_dataset, val_dataset = split_dataset(dataset, split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    model.eval()
    val_loss = 0.0
    predictions = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        pred = model(x)
        loss = criterion(pred, y)
        val_loss += loss.item() * x.size(0)
        predictions.append((pred.item(), y.item()))

    return train_loss / len(train_dataset), val_loss / len(val_dataset), predictions

def main():
    split_dir = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
    batch_size = 32
    hidden_dim = 256
    num_epochs = 10
    learning_rate = 1e-3

    split_files = sorted(glob.glob(os.path.join(split_dir, "splited_data_*.txt")))
    example_file = split_files[0]
    first_sample = next(iter(PerFileLSTMDataset(example_file)))
    input_dim = first_sample[0].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        random.shuffle(split_files)

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        all_predictions = []

        for filepath in tqdm(split_files, desc="🔧 Processing"):
            train_loss, val_loss, preds = train_and_validate_on_split_file(
                model, filepath, optimizer, criterion, device, batch_size)
            epoch_train_loss += train_loss
            epoch_val_loss += val_loss
            all_predictions.extend(preds)

        avg_train_loss = epoch_train_loss / len(split_files)
        avg_val_loss = epoch_val_loss / len(split_files)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"✅ Train Loss: {avg_train_loss:.6f} | 🧪 Val Loss: {avg_val_loss:.6f}")

        print("\n🔍 Sample Validation Predictions (First 100):")
        for i, (pred, truth) in enumerate(all_predictions[:100]):
            print(f"#{i:03d} → Prediction: {pred:.2f} | Ground Truth: {truth:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_dnn_sum_model.pt")
            print("💾 Best model saved to best_dnn_sum_model.pt")

        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss", marker='o')
    plt.plot(val_loss_history, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Over Epochs (Sum Prediction)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_sum.png")
    print("📉 Loss curve saved to loss_curve_sum.png")

if __name__ == "__main__":
    main()
