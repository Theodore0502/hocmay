import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_PATH = os.path.join(BASE_DIR, "features", "features.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")