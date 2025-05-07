import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as NetCDFDataset
import re
import glob
import argparse


def extract_timestamp(filename):
    match = re.search(r"(\d{8}_\d{4})", filename)
    return match.group(1) if match else "unknown"

def categorize_file(filename):
    if "Emissions" in filename:
        return "Emissions"
    elif "StateChm" in filename:
        return "StateChm"
    elif "StateMet" in filename:
        return "StateMet"
    elif "Restart" in filename:
        return "Restart"
    return None

def find_matching_files(input_dirs, answer_dir, time_format="%Y%m%d_%H%M"):
    print("[find_matching_files] Start", flush=True)
    input_files = {}
    answer_files = {}

    print("Collecting input files...", flush=True)
    for folder in input_dirs:
        for filepath in glob.glob(os.path.join(folder, "*.nc4")):
            filename = os.path.basename(filepath)
            timestamp = extract_timestamp(filename)
            if timestamp:
                category = categorize_file(filename)
                if category:
                    if timestamp not in input_files:
                        input_files[timestamp] = {}
                    input_files[timestamp][category] = filepath

    print("Collecting answer files...", flush=True)
    for filepath in glob.glob(os.path.join(answer_dir, "*.nc4")):
        filename = os.path.basename(filepath)
        timestamp = extract_timestamp(filename)
        if timestamp and "KppTotSteps" in filename:
            answer_files[timestamp] = filepath

    matched_pairs = []
    for timestamp, files in input_files.items():
        if timestamp in answer_files and all(key in files for key in ["Emissions", "StateMet", "Restart", "StateChm"]):
            matched_pairs.append((files, answer_files[timestamp]))
            print(f"Matched Pair - Timestamp: {timestamp}", flush=True)
            for key in ["Emissions", "StateMet", "Restart", "StateChm"]:
                print(f"  Input {key}: {files[key]}", flush=True)
            print(f"  Answer: {answer_files[timestamp]}", flush=True)

    print("[find_matching_files] Done", flush=True)
    return matched_pairs

def split_and_save_data_one_by_one(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for input_paths, output_path in dataset:
        timestamp = extract_timestamp(output_path)
        split_file_path = os.path.join(output_dir, f"splited_data_{timestamp}.txt")
        variable_name_path = os.path.join(output_dir, f"variable_names_{timestamp}.txt")
        if os.path.exists(split_file_path):
            print(f"[split_and_save_data] Already exists: {split_file_path}", flush=True)
            continue

        input_vectors = [[] for _ in range(6 * 24 * 24)]
        global_inputs = []
        input_var_names = []
        global_var_names = []
        output_var_names = []

        for key, path in input_paths.items():
            try:
                with NetCDFDataset(path, "r") as ds:
                    for var_name, var in ds.variables.items():
                        print(f"[split_and_save_data] Processing input variable '{var_name}' from {path}, shape={var.shape}", flush=True)
                        arr = var[:]
                        if not isinstance(arr, np.ndarray) or arr.size == 0 or arr.ndim < 3:
                            print(f"[split_and_save_data] Skipping variable '{var_name}' due to insufficient dimensions", flush=True)
                            continue

                        mean, std = np.mean(arr), np.std(arr)
                        if std != 0:
                            arr = (arr - mean) / std

                        shape = arr.shape
                        if len(shape) >= 3 and shape[-3:] == (6, 24, 24):
                            arr = arr.reshape((-1, 6, 24, 24))
                            input_var_names.append(var_name)
                            for c in range(6):
                                for i in range(24):
                                    for j in range(24):
                                        idx = c * 576 + i * 24 + j
                                        input_vectors[idx].append(arr[:, c, i, j].flatten())
                        elif shape == (72, 144, 24):
                            try:
                                arr = arr.reshape((72, 6, 24, 24))
                                input_var_names.append(var_name)
                                for c in range(6):
                                    for i in range(24):
                                        for j in range(24):
                                            idx = c * 576 + i * 24 + j
                                            input_vectors[idx].append(arr[:, c, i, j].flatten())
                            except Exception as e:
                                print(f"[split_and_save_data] Failed to reshape (72,144,24) to (72,6,24,24) for '{var_name}': {e}", flush=True)
                                continue
                        else:
                            global_inputs.append(arr.flatten())
                            global_var_names.append(var_name)
            except Exception as e:
                print(f"Error loading input {path}: {e}", flush=True)

        global_flat = np.concatenate(global_inputs) if global_inputs else np.array([])

        try:
            with NetCDFDataset(output_path, "r") as ds:
                output_vectors = [None] * (6 * 24 * 24)
                for var_name, var in ds.variables.items():
                    if var_name != "KppTotSteps":
                        continue
                    print(f"[split_and_save_data] Processing output variable '{var_name}' from {output_path}, shape={var.shape}", flush=True)
                    arr = var[:]
                    print(f"[debug] Output variable '{var_name}' loaded. Shape: {arr.shape}, dtype: {arr.dtype}, size: {arr.size}", flush=True)
                    if not isinstance(arr, np.ndarray) or arr.size == 0:
                        print(f"[debug] Skipping variable '{var_name}' due to empty or invalid data", flush=True)
                        continue

                    arr = arr.reshape((-1, 6, 24, 24))
                    output_var_names.append(var_name)
                    for c in range(6):
                        for i in range(24):
                            for j in range(24):
                                idx = c * 576 + i * 24 + j
                                if output_vectors[idx] is None:
                                    output_vectors[idx] = []
                                output_vectors[idx].append(arr[:, c, i, j].flatten())

                with open(split_file_path, "w") as f:
                    for idx in range(6 * 24 * 24):
                        x_chunks = input_vectors[idx]
                        y_chunks = output_vectors[idx]
                        if not x_chunks or not y_chunks:
                            continue
                        x = np.concatenate(x_chunks + ([global_flat] if global_flat.size > 0 else []))
                        y = np.concatenate(y_chunks)
                        x_str = ",".join(map(str, x.tolist()))
                        y_str = ",".join(map(str, y.tolist()))
                        f.write(f"{x_str}||{y_str}\n")
            print(f"[split_and_save_data] ✅ Finished processing timestamp {timestamp}. Output saved to: {split_file_path}", flush=True)

            # NEW: Save variable names to file
            with open(variable_name_path, "w") as vf:
                vf.write("Input Variables (chunked):\n")
                for name in sorted(set(input_var_names)):
                    vf.write(f"{name}\n")
                vf.write("\nGlobal Input Variables:\n")
                for name in sorted(set(global_var_names)):
                    vf.write(f"{name}\n")
                vf.write("\nOutput Variables:\n")
                for name in sorted(set(output_var_names)):
                    vf.write(f"{name}\n")
            print(f"[split_and_save_data] 📝 Saved variable names to: {variable_name_path}", flush=True)

        except Exception as e:
            print(f"Error loading output {output_path}: {e}", flush=True)


class LSTMDataset(Dataset):
    def __init__(self, dataset, split_dir):
        print("[LSTMDataset.__init__] Loading split data", flush=True)
        self.samples = []
        for _, output_path in dataset:
            timestamp = extract_timestamp(output_path)
            split_file_path = os.path.join(split_dir, f"splited_data_{timestamp}.txt")
            if not os.path.exists(split_file_path):
                print(f"[LSTMDataset] Missing split file: {split_file_path}", flush=True)
                continue
            with open(split_file_path, "r") as f:
                for line in f:
                    x_str, y_str = line.strip().split("||")
                    x = np.fromstring(x_str, sep=",")
                    y = np.fromstring(y_str, sep=",")
                    self.samples.append((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)))
        print(f"[LSTMDataset.__init__] Loaded {len(self.samples)} total samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs='+', required=True)
    parser.add_argument("--answer_dir", required=True)
    parser.add_argument("--split_dir", default="/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split")
    parser.add_argument("--do_split", action="store_true", help="Perform splitting and saving if set")
    args = parser.parse_args()

    dataset_pairs = find_matching_files(args.input_dirs, args.answer_dir)

    if args.do_split:
        for pair in dataset_pairs:
            split_and_save_data_one_by_one([pair], args.split_dir)

    dataset = LSTMDataset(dataset_pairs, args.split_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for x, y in loader:
        print(x.shape, y.shape)
        break

if __name__ == "__main__":
    main()
