import numpy as np
import pandas as pd

data = np.load("train.npz", allow_pickle=True)

def save_npz_to_csv(npz_data, key, filename):
    if key in npz_data.files:
        pd.DataFrame(npz_data[key]).to_csv(filename, index=False)
        print(f"{filename} has been created")
    else:
        print(f"Error: Key '{key}' not found in the NPZ file.")

def preview_npz_array(npz_data, key):
    if key in npz_data.files:
        df = pd.DataFrame(npz_data[key])
        print(f"\nPreview of '{key}':")
        print(df.head(3))  
    else:
        print(f"Error: Key '{key}' not found in the NPZ file.")

#save_npz_to_csv(data, 'X_train', 'X_train.csv')
#save_npz_to_csv(data, 'y_train', 'y_train.csv')

preview_npz_array(data, 'X_train')


