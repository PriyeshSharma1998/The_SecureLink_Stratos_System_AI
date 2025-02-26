import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Paths
DATA_PATH = os.getenv("DATA_PATH")
LOG_PATH = os.getenv("LOG_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
PREPROCESSING_PATH = os.getenv("PREPROCESSING_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# Output Filenames
OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME")
OUTPUT_METADATAFILENAME = os.getenv("OUTPUT_METADATAFILENAME")

# API Key
API_KEY = os.getenv("API_KEY")

# Target Feature Name
TARGET_FEATURE_NAME = os.getenv("TARGET_FEATURE_NAME")

# Logging Level
LOG_LEVEL = os.getenv("LOG_LEVEL")

# KNN Model Hyperparameters
KNN_PARAMS = {
    "n_neighbors": int(os.getenv("KNN_N_NEIGHBORS")),
    "weights": os.getenv("KNN_WEIGHTS"),
    "algorithm": os.getenv("KNN_ALGORITHM"),
    "leaf_size": int(os.getenv("KNN_LEAF_SIZE")),
    "p": int(os.getenv("KNN_P")),
}

# Random Forest Model Hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": int(os.getenv("RF_N_ESTIMATORS")),
    "criterion": os.getenv("RF_CRITERION"),
    "max_depth": None if os.getenv("RF_MAX_DEPTH") == "None" else int(os.getenv("RF_MAX_DEPTH")),
    "min_samples_split": int(os.getenv("RF_MIN_SAMPLES_SPLIT")),
    "min_samples_leaf": int(os.getenv("RF_MIN_SAMPLES_LEAF")),
    "bootstrap": os.getenv("RF_BOOTSTRAP") == "True",
    "random_state": None if os.getenv("RF_RANDOM_STATE") == "None" else int(os.getenv("RF_RANDOM_STATE")),
}

# XGBoost Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS")),
    "max_depth": int(os.getenv("XGB_MAX_DEPTH")),
    "learning_rate": float(os.getenv("XGB_LEARNING_RATE")),
    "subsample": float(os.getenv("XGB_SUBSAMPLE")),
    "colsample_bytree": float(os.getenv("XGB_COLSAMPLE_BYTREE")),
    "random_state": int(os.getenv("XGB_RANDOM_STATE")),
}

# Logistic Model Hyperparameters
LOGISTIC_PARAMS = {
    "loss": os.getenv("LOGISTIC_LOSS"),
    "penalty": os.getenv("LOGISTIC_PENALTY"),
    "alpha": float(os.getenv("LOGISTIC_ALPHA")),
    "learning_rate": os.getenv("LOGISTIC_LEARNING_RATE"),
    "max_iter": int(os.getenv("LOGISTIC_MAX_ITER")),
    "random_state": None if os.getenv("LOGISTIC_RANDOM_STATE") == "None" else int(os.getenv("LOGISTIC_RANDOM_STATE")),
}
