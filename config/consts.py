# FILE TO KEEP STATIC CONSTS #

# --------------------------------------------------
# NPZ FILE PATHS
# --------------------------------------------------
TRAIN_NPZ_PATH = "data/train.npz"
TEST_NPZ_PATH  = "data/test.npz"

# GZIP PATHS
TRAIN_GZ_PATH  = "data/train.npz.gz"
TEST_GZ_PATH   = "data/test.npz.gz"

# --------------------------------------------------
# GOOGLE DRIVE FILE IDS (to be provided by the user)
# --------------------------------------------------
TRAIN_DRIVE_ID = "1dcLpdu0As6fbA7wCSn5CFb0STDEyjR74"
TEST_DRIVE_ID  = "13689UKuknqndsW5QjlPA-Fbw5eF6HlQj"


# --------------------------------------------------
# CSV FILE PATHS
# --------------------------------------------------
METADATA_TRAIN_PATH = "data/metadata_train.csv"
METADATA_TEST_PATH  = "data/metadata_test.csv"

# --------------------------------------------------
# NPZ KEYS
# --------------------------------------------------
X_TRAIN_KEY = "X_train"
Y_TRAIN_KEY = "y_train"
X_TEST_KEY  = "X_test"
IDS_KEY     = "ID"

# --------------------------------------------------
# LOGGING & OUTPUT
# --------------------------------------------------
LOG_FILE_PREFIX = "pipeline"
OUTPUT_DIR = "outputs"

# --------------------------------------------------
# MODEL TAGS
# --------------------------------------------------
LOGISTIC_REGRESSION = "logistic_regression"
SVM = "svm"
LINEAR_SVM = "linear_svm"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
STACK = "stack"
RANDOMIZED_TREE = "randomized_tree"

# --------------------------------------------------
# SEARCH MODE
# --------------------------------------------------
RANDOM_MODE = "random"
GRID_SEARCH_MODE = "grid"
OPTUNA_MODE = "optuna"

# --------------------------------------------------
# SCORING METRIC
# --------------------------------------------------
DEFAULT_SCORING_METRIC = "f1_macro"
