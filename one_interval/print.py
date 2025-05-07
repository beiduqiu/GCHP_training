import os
import glob

def compare_lines(line1, line2):
    try:
        x1 = line1.split("||")[0].strip()
        x2 = line2.split("||")[0].strip()
        arr1 = [float(val) for val in x1.split(",")]
        arr2 = [float(val) for val in x2.split(",")]
    except Exception:
        return None  # skip malformed lines

    if len(arr1) != len(arr2):
        return None

    same_count = sum(1 for a, b in zip(arr1, arr2) if abs(a - b) < 1e-6)
    total = len(arr1)
    similarity = same_count / total if total > 0 else 0
    return same_count, total, similarity

def print_comparison_to_first_line(directory, max_total_lines=60, max_lines_per_file=30):
    total_printed = 0
    files = glob.glob(os.path.join(directory, "*"))

    for file_path in sorted(files):
        if total_printed >= max_total_lines:
            break

        if os.path.isfile(file_path):
            print(f"\n===== File: {os.path.basename(file_path)} =====")
            with open(file_path, "r") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines_per_file or total_printed >= max_total_lines:
                        break
                    lines.append(line.strip())
                    total_printed += 1

                if len(lines) < 2:
                    continue

                base_line = lines[0]
                for i in range(1, len(lines)):
                    res = compare_lines(base_line, lines[i])
                    if res is not None:
                        same_count, total, sim = res
                        print(f"Line 0 vs Line {i}: {same_count}/{total} same ({sim:.2%})")

def main():
    input_dir = "/storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/split"
    if not os.path.exists(input_dir):
        print(f"Directory does not exist: {input_dir}")
        return
    print_comparison_to_first_line(input_dir)

if __name__ == "__main__":
    main()
