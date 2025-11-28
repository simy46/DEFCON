# FILE TO KEEP STATIC CONSTS #

# NPZ FILES PATH
TRAIN_NPZ_PATH = 'data/train.npz'
TEST_NPZ_PATH = 'data/test.npz'
# GZIP
TRAIN_GZ_PATH  = "data/train.npz.gz"
TEST_GZ_PATH  = "data/test.npz.gz"


# CSV FILES PATH
METADATA_TRAIN_PATH = 'data/metadata_train.csv'
METADATA_TEST_PATH = 'data/metadata_test.csv'


# KEYS
X_TRAIN_KEY = 'X_train'
Y_TRAIN_KEY = 'y_train'
X_TEST_KEY  = "X_test"
IDS_KEY     = "ID"


# LOGS FILE PREFIX
LOG_FILE_PREFIX = "pipeline"

# OUTPUTS DIR
OUTPUT_DIR = "outputs"

# MODELS
LOGISTIC_REGRESSION = 'logistic_regression'
SVM = 'svm'
RANDOM_FOREST = 'random_forest'
XGBOOST = 'xgboost'
STACK = 'stack'


# SEARCH MODE
RANDOM_MODE = 'random'
GRID_SEARCH_MODE = 'grid'
OPTUNA_MODE = 'optuna'

# SCORE METRIC
DEFAULT_SCORING_METRIC = 'f1_macro'