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
### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\Activate
```

## 4. Dependencies
```bash
pip install -r requirements.txt
```

## 5. Config file structure
Each config file has three parts: `model`, `preprocessing`, and `hyperoptimization`.

### 5.1. model
Defines models base hyperparameters.  
When running hyperopt:
- parameters set to `None` fall back to the values here 
- the final `*_best.yaml` overwrites this section with optimized values

### 5.2. preprocessing
Controls all preprocessing steps.  
Each step has `enabled: true/false`.  
Only steps with `enabled: true` are applied.  
This structure is identical for every model.

### 5.3. hyperoptimization
Enables hyperparameter search (`grid`, `random`, or `optuna`).  
Only parameters defined here are searched.  
Parameters set to `None` keep their value from the `model` section.  

## 1. Running the main pipeline
The pipeline is controlled through a YAML config file.
```bash
python main.py --config config/model_rt_best.yaml
```

Other available configs are listed on /config folder. 
The command that we gave is to run the same configs as our submission.

## 2. What the script does
1. Loads the YAML config
2. Loads training and test data
3. Applies preprocessing (scaling, PCA, etc.)
4. Trains the model defined in the config
5. Generates predictions on the test set
6. Saves a submission file in submissions/submission_<timestamp>.csv

## 3. Output
```bash
submissions/submission_2025-01-04_14-32-10.csv
```

## 1. Run hyperparameter search
This script searches for the best hyperparameters and generates a new optimized config file.
```bash
python find_best_params.py --config config/model_<name>_best.yaml
```
Other available configs are listed on /config folder. 

## 2. What the script does
1. Loads the YAML config
2. Loads training and test data
3. Applies preprocessing (same pipeline as the main script)
4. Builds the model and its search space
5. Runs hyperparameter search (Grid Search, Random Search, or Optuna)
6. Logs the best score and best parameters
7. Writes a new file: config/model_rt_best.yaml with the updated hyperparameters

## 3. Output
```bash
config/model_<name>_best.yaml
```

After generating this file, use it on the main.py pipeline
```bash
python main.py --config config/model_<name>_best.yaml
```


