import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

class PerFileLSTMDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, "r") as f:
            for line in f:
                if "||" not in line:
                    continue
                x_str, y_str = line.strip().split("||")
                x = np.fromstring(x_str, sep=",")
                y = np.fromstring(y_str, sep=",").sum(keepdims=True)  # Predict sum
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Predict scalar sum

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_on_split_file(model, filepath, optimizer, criterion, device, batch_size, print_buffer):
    dataset = PerFileLSTMDataset(filepath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        pred = model(x)

        # Store up to 200 predictions
        if len(print_buffer) < 200:
            for i in range(min(pred.shape[0], 200 - len(print_buffer))):
                print_buffer.append((y[i].item(), pred[i].item()))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        print_buffer = []

        for filepath in tqdm(train_files, desc="🔧 Training"):
            file_loss = train_on_split_file(model, filepath, optimizer, criterion, device, batch_size, print_buffer)
            epoch_loss += file_loss
        avg_train_loss = epoch_loss / len(train_files)

        val_loss = 0.0
        for filepath in val_files:
            val_loss += validate_on_file(model, filepath, criterion, device, batch_size)
        avg_val_loss = val_loss / len(val_files)

        print(f"✅ Train Loss: {avg_train_loss:.6f} | 🧪 Val Loss: {avg_val_loss:.6f}")

        print("\n📊 Sample predictions (Ground Truth vs Prediction):")
        for truth, pred in print_buffer:
            print(f"GT: {truth:.4f} | Pred: {pred:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_sum_model.pt")
            print("💾 Best model saved to best_sum_model.pt")

if __name__ == "__main__":
    main()
