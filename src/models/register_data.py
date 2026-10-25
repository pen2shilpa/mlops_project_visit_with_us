import os
import pandas as pd
import logging

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FILE_PATH = "tourism_project/data/tourism.csv"
assert os.path.exists(FILE_PATH), f"Data file not found at {FILE_PATH}"

# Load the raw dataset
df = pd.read_csv(FILE_PATH)

# Remove the first unnamed column
df.drop(columns=[df.columns[0]], inplace=True)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

missing = [col for col in expected_columns if col not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

logger.info("Tourism Dataset registered successfully.")
logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
logger.info("Columns Names: %s", list(df.columns))

logger.info("Distribution of Tourism package purchased by customers")
logger.info(df["ProdTaken"].value_counts())
