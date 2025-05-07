import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import glob
from tqdm import tqdm
import random

# ===== Dataset =====
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

# ===== Model =====
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ===== Validation Sampler =====
@torch.no_grad()
def sample_and_validate_all(model, file_list, device, sample_per_file=200):
    model.eval()
    total_mse = 0.0
    total_sum_mse = 0.0
    total_count = 0

    for file_idx, filepath in enumerate(file_list):
        dataset = PerFileLSTMDataset(filepath)
        if len(dataset) == 0:
            continue

        indices = random.sample(range(len(dataset)), min(sample_per_file, len(dataset)))
        sampled_subset = Subset(dataset, indices)
        loader = DataLoader(sampled_subset, batch_size=1, shuffle=False)

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            pred = model(x)

            pred_np = pred.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()

            pred_sum = pred_np.sum()
            y_sum = y_np.sum()

            # Print results
            print(f"\n📁 File {file_idx+1}: {os.path.basename(filepath)} | Sample {i+1}")
            print("   📐 Prediction:", np.array2string(pred_np, precision=5, separator=","))
            print("   🎯 Target:    ", np.array2string(y_np, precision=5, separator=","))
            print(f"   🔍 Sum(pred) = {pred_sum:.5f}, Sum(target) = {y_sum:.5f}")

            # Accumulate loss
            sample_mse = ((pred_np - y_np) ** 2).mean()
            sum_mse = (pred_sum - y_sum) ** 2

            total_mse += sample_mse
            total_sum_mse += sum_mse
            total_count += 1

    # Final average
    if total_count > 0:
        avg_mse = total_mse / total_count
        avg_sum_mse = total_sum_mse / total_count
        print(f"\n✅ Average MSE over all 1x72 predictions: {avg_mse:.6f}")
        print(f"✅ Average MSE of the prediction sums:    {avg_sum_mse:.6f}")
    else:
        print("⚠️ No valid samples found for validation.")

# ===== Main Entry =====
def main():
    model_path = "best_model11.pt"
    input_dir = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
    sample_per_file = 200
    hidden_dim = 256

    file_list = sorted(glob.glob(os.path.join(input_dir, "splited_data_*.txt")))
    if len(file_list) == 0:
        print("❌ No data files found.")
        return

    # Infer input/output dimension
    example_sample = next(iter(PerFileLSTMDataset(file_list[0])))
    input_dim = example_sample[0].shape[0]
    output_dim = example_sample[1].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_dim, hidden_dim, output_dim).to(device)

    # Load model
    print(f"📥 Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Validate across all files
    sample_and_validate_all(model, file_list, device, sample_per_file=sample_per_file)

if __name__ == "__main__":
    main()
