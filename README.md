# GCHP Workload Prediction with Deep Neural Networks

This project predicts the computational workload for GCHP (GEOS-Chem High Performance) using a Deep Neural Network (DNN). Accurate workload estimation helps improve dynamic load balancing during large-scale atmospheric simulations.

## 📦 Project Structure

```
.
├── DATA process/                      # Raw and processed NetCDF & log data
├── DNN/                      # Model training and evaluation code
└── README.md                 # This documentation
```

---

## 🛠 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourname/GCHP_training.git
cd GCHP_training
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure `PyTorch`, `netCDF4`, `numpy`, `tqdm`, and `matplotlib` are installed.

---

## 📊 Data Preprocessing

### 1. Prepare input folders

Run simulation once, set up the diagnostic frequency to be 20 minutes (or any other frequency that you want). Save this output to be a output folder.

```
output/
├── Emissions/
├── StateMet/
├── StateChm/
└── Restart/
```

Each file should include a timestamp in the name, e.g., `20190701_0000z_Emissions.nc4`.

### 2. Run the preprocessing script

```bash
cd DATA process
sh data_split.sh #you will need to modify the corresponding directories
```

This script will:
- Parse all `.nc4` files
- Match by timestamp
- Merge features and workload labels into txt files

---

## 🧠 Model Training

### Train the DNN model

```bash
cd DNN
bsub < training.bsub
```

The best model will be saved as `models/best_dnn_model.pt`.

---



## ✅ Notes

- Input files with fewer than 3456 rows will be skipped.
- Intermediate results (columns and predictions) are written incrementally for robustness.
- Model can be extended to multi-timestep or regional inference.

---

## 📬 Contact

Maintainer: Zifan Wang  
Email: w.zifan1@wustl.edu 
For issues, open a GitHub issue or contact the maintainer.
