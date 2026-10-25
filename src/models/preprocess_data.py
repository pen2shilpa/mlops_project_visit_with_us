import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FILE_PATH = "data/tourism.csv"
assert os.path.exists(FILE_PATH), f"Data file not found at {FILE_PATH}"

df = pd.read_csv(FILE_PATH, on_bad_lines="skip")
df_original = df.copy()

logger.info(f"STEP :: Data is read from the file successfully")

# Check for any null or missing values
missing_values_count = df.isna().sum().sum()

if missing_values_count > 0:
    logger.warning(f"Dataset contains {missing_values_count} missing values.")

    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].to_dict()
    raise ValueError(
        f"Pipeline halted: {missing_values_count} missing values detected across columns: {list(cols_with_missing.keys())}"
    )

# Check for any duplicates
duplicates_count = df.duplicated().sum()

if duplicates_count > 0:
    logger.warning(f"Dataset contains {duplicates_count} duplicate rows. Dropping them.")
    df.drop_duplicates(inplace=True)

# Drop unnecessary columns
df.drop(columns=["CustomerID"], inplace=True)

# Handle data anomalies in Gender and MaritalStatus
df.replace({
    "Gender": {"Fe Male": "Female"},
    "MaritalStatus": {"Single": "Unmarried"}
}, inplace=True)

logger.info(f"STEP :: Data preprocessing is completed successfully")

# Create features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratify to handle imbalanced classes
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally for artifacts
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

logger.info(f"STEP :: Data preparation Test/Train data split is completed successfully.")
