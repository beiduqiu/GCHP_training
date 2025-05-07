import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import csv
from tqdm import tqdm
import pandas as pd

class PerFileLSTMDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, "r") as f:
            for line in f:
                if "||" not in line:
                    continue
                x_str, _ = line.strip().split("||")
                x = np.fromstring(x_str, sep=",")
                self.samples.append(torch.tensor(x, dtype=torch.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout=0.2):
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

def predict_sum_for_file(model, filepath, device, batch_size=32):
    dataset = PerFileLSTMDataset(filepath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sums = []
    for x in loader:
        x = x.to(device).unsqueeze(1)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        batch_sums = np.sum(pred, axis=1)
        sums.extend(batch_sums.tolist())

    return sums

def append_column_to_csv(csv_path, column_name, column_data):
    df = pd.DataFrame(column_data, columns=[column_name])

    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        existing_df = pd.read_csv(csv_path)
        if len(existing_df) != len(df):
            raise ValueError(f"Row mismatch: existing file has {len(existing_df)} rows, new column has {len(df)}")
        combined_df = pd.concat([existing_df, df], axis=1)
        combined_df.to_csv(csv_path, index=False)

def main():
    split_dir = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
    model_path = "DNN2.pt"
    output_csv = "prediction_sum_matrix.csv"

    print("🔍 Scanning files...")
    split_files = sorted(glob.glob(os.path.join(split_dir, "splited_data_*.txt")))
    print(f"📁 Found {len(split_files)} files.\n")

    print("🧠 Initializing model...")
    sample_x = next(iter(PerFileLSTMDataset(split_files[0])))
    input_dim = sample_x.shape[0]
    output_dim = 72

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(input_dim=input_dim, output_dim=output_dim).to(device)

    print("📥 Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}\n")

    print(f"🚀 Starting per-file prediction and CSV write...\n")
    for filepath in tqdm(split_files, desc="📂 Processing files"):
        start = time.time()
        filename = os.path.basename(filepath)
        try:
            sums = predict_sum_for_file(model, filepath, device)
            append_column_to_csv(output_csv, filename, sums)
            print(f"✅ {filename} done in {time.time() - start:.2f} sec")
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")

    print("\n🎉 All files processed. Final CSV saved to:", output_csv)

if __name__ == "__main__":
    main()
