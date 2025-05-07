import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
                y = np.fromstring(y_str, sep=",")
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout=0.05):
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

def train_on_split_file(model, filepath, optimizer, criterion, device, batch_size, print_flag):
    dataset = PerFileLSTMDataset(filepath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        pred = model(x)

        if not print_flag[0]:
            print(f"\n📂 File: {filepath}")
            print("📐 y.shape:", y.shape)
            print("📐 pred.shape:", pred.shape)
            for i in range(min(5, y.shape[0])):
                print(f"🔢 y[{i}]:", y[i].cpu().numpy())
                print(f"🔢 pred[{i}]:", pred[i].detach().cpu().numpy())
            print_flag[0] = True

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

    return running_loss / len(dataset) if len(dataset) > 0 else 0.0

@torch.no_grad()
def validate_on_file(model, filepath, criterion, device, batch_size):
    dataset = PerFileLSTMDataset(filepath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        pred = model(x)
        loss = criterion(pred, y)
        running_loss += loss.item() * x.size(0)

    return running_loss / len(dataset) if len(dataset) > 0 else 0.0

@torch.no_grad()
def evaluate_predictions(model, file_list, device, max_samples=200):
    model.eval()
    all_preds = []
    all_labels = []

    for filepath in file_list:
        dataset = PerFileLSTMDataset(filepath)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            pred = model(x)
            all_preds.append(pred.squeeze(0).cpu().numpy())
            all_labels.append(y.squeeze(0).cpu().numpy())

            if len(all_preds) >= max_samples:
                break
        if len(all_preds) >= max_samples:
            break

    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)

    mse_full = np.mean((all_preds - all_labels) ** 2)
    mse_sum = np.mean((np.sum(all_preds, axis=1) - np.sum(all_labels, axis=1)) ** 2)

    print(f"\n📊 MSE (vector): {mse_full:.6f} | MSE (sum): {mse_sum:.6f}")
    print("🔍 Sample comparisons:")
    for i in range(min(max_samples, len(all_preds))):
        print(f"🔢 pred[{i}]:", np.round(all_preds[i], 2))
        print(f"🔢 true[{i}]:", np.round(all_labels[i], 2))
        print(f"    sum(pred): {all_preds[i].sum():.2f} | sum(true): {all_labels[i].sum():.2f}\n")

def main():
    split_dir = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
    batch_size = 32
    hidden_dim = 256
    num_epochs = 10
    learning_rate = 1e-3

    split_files = sorted(glob.glob(os.path.join(split_dir, "splited_data_*.txt")))
    num_total = len(split_files)
    num_train = int(num_total * 0.9)
    train_files = split_files[:num_train]
    val_files = split_files[num_train:]

    example_file = train_files[0]
    first_sample = next(iter(PerFileLSTMDataset(example_file)))
    input_dim = first_sample[0].shape[0]
    output_dim = first_sample[1].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        random.shuffle(train_files)

        epoch_loss = 0.0
        print_flag = [False]

        for filepath in tqdm(train_files, desc="🔧 Training"):
            file_loss = train_on_split_file(model, filepath, optimizer, criterion, device, batch_size, print_flag)
            epoch_loss += file_loss
        avg_train_loss = epoch_loss / len(train_files)
        train_loss_history.append(avg_train_loss)

        val_loss = 0.0
        for filepath in val_files:
            val_loss += validate_on_file(model, filepath, criterion, device, batch_size)
        avg_val_loss = val_loss / len(val_files)
        val_loss_history.append(avg_val_loss)

        print(f"✅ Train Loss: {avg_train_loss:.6f} | 🧪 Val Loss: {avg_val_loss:.6f}")
        evaluate_predictions(model, val_files, device, max_samples=200)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_dnn_model.pt")
            print("💾 Best model saved to best_dnn_model.pt")

        scheduler.step()

    # 📈 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss", marker='o')
    plt.plot(val_loss_history, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("📊 Loss curve saved to loss_curve.png")

if __name__ == "__main__":
    main()
