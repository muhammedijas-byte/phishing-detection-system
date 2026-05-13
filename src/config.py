from pathlib import Path

# Base directory of your project (folder that contains src, data, models, etc.)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folder and default CSV path
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "phishing_data.csv"  # we'll put the dataset here later

# Models folder
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Name of the target/label column in the dataset
# If your dataset uses a different name (like 'Result'), we'll change this later.
TARGET_COL = "status"




# Random seed for reproducibility
RANDOM_STATE = 42
