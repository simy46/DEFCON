# DEFCON
This repository contains a full machine-learning pipeline for training a model, generating predictions, and producing a `submission.csv` file for Kaggle.

## 1. Requirements

- Python 3.9+
- Git
- A terminal (Linux, macOS, or Windows PowerShell)

## 2. Clone the repository

```bash
git clone https://github.com/simy46/DEFCON.git
cd DEFCON
```

## 3. Venv
# Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

# Windows
```bash
python -m venv venv
venv\Scripts\Activate
```

## 4. Dependencies
```bash
pip install -r requirements.txt
```

## 5. Running the pipeline
The pipeline is controlled through a YAML config file.
```bash
python main.py --config config/model_rt_best.yaml
```

Other available configs are listed on /config folder. The command that we gave is to run the same configs as our submission.

## 6. What the script does
1. Loads the YAML config
2. Loads training and test data
3. Applies preprocessing (scaling, PCA, etc.)
4. Trains the model defined in the config
5. Generates predictions on the test set
6. Saves a submission file in submissions/submission_<timestamp>.csv

## 7. Output
```bash
submissions/submission_2025-01-04_14-32-10.csv
```