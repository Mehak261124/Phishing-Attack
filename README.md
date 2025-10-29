# 🧠 Phishing Attack Detection

This project aims to detect phishing or malicious network traffic using machine learning models trained on the CICIDS 2017 dataset. It includes exploratory data analysis, a baseline model, and a neural network for classification.

---

## 📂 Dataset Setup

1. Go to the official CICIDS 2017 Dataset page:  
   👉 https://www.unb.ca/cic/datasets/ids-2017.html

2. Download the file `GeneratedLabelledFlows.zip`.

3. Extract the contents and place the folder (e.g., `TrafficLabelling/`) inside your project directory.

**Example structure:**
```
Phishing Attack/
├── data_explore.py
├── baseline_model.py
├── neural_network.py
├── requirements.txt
├── .gitignore
└── TrafficLabelling/
    ├── Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
    ├── Monday-WorkingHours.pcap_ISCX.csv
    └── ...
```

---

## ⚙️ Environment Setup

Make sure you have Python 3.12.3 (or newer).

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Run the scripts in order for a complete workflow:

### 1. Data exploration
```bash
python data_explore.py
```

### 2. Baseline model
```bash
python baseline_model.py
```

### 3. Neural network model
```bash
python neural_network.py
```

---

## 📊 Project Overview

| Script | Description |
|--------|-------------|
| `data_explore.py` | Performs exploratory data analysis and preprocessing on the CICIDS dataset |
| `baseline_model.py` | Implements a traditional ML baseline model for classification |
| `neural_network.py` | Builds and trains a deep neural network for phishing attack detection |

---

## 🧾 Notes

* The dataset is not included in this repository due to its large size.
* Ensure that the extracted dataset folder (e.g., `TrafficLabelling/`) is in the same directory as your Python scripts.
* If you face memory issues, consider using a subset of the data for local experiments.

---


