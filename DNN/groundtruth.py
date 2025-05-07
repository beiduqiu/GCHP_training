import os
import glob
import numpy as np
from tqdm import tqdm
import csv

REQUIRED_ROWS = 3456

def generate_ground_truth_workload_incremental(input_dir, output_file):
    input_files = sorted(glob.glob(os.path.join(input_dir, "*")))
    temp_columns = []

    print(f"🔍 Found {len(input_files)} files. Processing incrementally...")

    for file_path in tqdm(input_files, desc="Processing files", unit="file"):
        column = []
        with open(file_path, "r") as f:
            for line in f:
                if "||" not in line:
                    continue
                try:
                    _, y_str = line.strip().split("||")
                    y = np.fromstring(y_str, sep=",")
                    column.append(np.sum(y))
                except Exception as e:
                    print(f"⚠️ Error in {os.path.basename(file_path)}: {e}")

        if len(column) == REQUIRED_ROWS:
            temp_columns.append((os.path.basename(file_path), column))

            if not os.path.exists(output_file):
                # First column: create new file
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([os.path.basename(file_path)])
                    for val in column:
                        writer.writerow([val])
            else:
                # Append as new column
                with open(output_file, "r", newline="") as f:
                    existing_rows = list(csv.reader(f))

                if len(existing_rows) != REQUIRED_ROWS + 1:
                    print(f"❗ Row mismatch in {output_file}. Skipping append.")
                    continue

                new_rows = [existing_rows[0] + [os.path.basename(file_path)]]  # header row
                for i in range(REQUIRED_ROWS):
                    new_rows.append(existing_rows[i + 1] + [str(column[i])])

                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(new_rows)

        else:
            tqdm.write(f"❌ Skipped {os.path.basename(file_path)} (has {len(column)} rows)")

    print(f"\n✅ Output incrementally built at: {output_file}")

# === CONFIG ===
input_directory = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
output_csv = "/storage1/guerin/Active/geos-chem/ZifanRunDir/GCHP_training/DNN/groundtruth_workload.csv"

# === RUN ===
if __name__ == "__main__":
    generate_ground_truth_workload_incremental(input_directory, output_csv)
