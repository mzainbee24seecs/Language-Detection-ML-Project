# Quick Start Guide - Language Detection Project

## Setup (5 minutes)

### 1. Install Dependencies

```powershell
cd language-detection
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

Download from: https://www.kaggle.com/datasets/basilb2s/language-detection
- Place CSV in `data/raw/` folder
- Rename to `language_detection.csv`

### 3. Run Complete Pipeline

```powershell
python main.py --train-all
```

This will:
1. Preprocess the data (create TF-IDF features)
2. Train 4 classical ML models (SVM, Random Forest, Logistic Regression, Naive Bayes)
3. Train deep learning CNN model
4. Evaluate and compare all models
5. Generate visualizations in `results/figures/`

## Expected Results

- **Training Time:** 20-40 minutes total
- **Best Model:** Linear SVM (98.58% accuracy)
- **Results Location:** `results/metrics/final_comparison.csv`

## Step-by-Step (Optional)

```powershell
# Step 1: Preprocess
python main.py --preprocess

# Step 2: Train classical ML
python main.py --train-classical

# Step 3: Train deep learning
python main.py --train-dl --dl-model cnn

# Step 4: Evaluate
python main.py --evaluate
```

## View Results

```powershell
# View metrics table
type results\metrics\final_comparison.csv

# Open figures folder
explorer results\figures\
```

## Command Options

- `--train-all`: Run entire pipeline
- `--preprocess`: Preprocess data only
- `--train-classical`: Train classical ML models
- `--train-dl`: Train deep learning model
- `--evaluate`: Evaluate all trained models
- `--dl-model {cnn,bilstm,hybrid}`: Choose DL architecture
- `--epochs N`: Training epochs (default: 50)
- `--batch-size N`: Batch size (default: 64)

## Troubleshooting

### Out of memory
```powershell
python main.py --train-dl --batch-size 32
```

### Module not found
```powershell
pip install -r requirements.txt --upgrade
```

### Dataset not found
Ensure file is at: `data/raw/language_detection.csv`

---

**Expected Results:**
- SVM: ~98.6% accuracy
- Logistic Regression: ~98.5% accuracy
- CNN: ~98.0% accuracy
- All models: >97% accuracy
