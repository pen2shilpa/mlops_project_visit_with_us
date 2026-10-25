import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load files from artifacts
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

# Define numeric and categorical features
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Create column transformer
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# scale pos weight
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
# print(f"scale_pos_weight {scale_pos_weight}")

# Define XGBoost classifier
# As the dataset had imbalanced classes with only 797 prod taken out of 4128, auc is chosen as eval_metric whoch balances both precision and recall
# aucpr (Area Under the Precision-Recall Curve)
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='aucpr', scale_pos_weight=scale_pos_weight)

# Define the pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define the parameters
param_grid = {
    'xgbclassifier__n_estimators': [100, 150],
    'xgbclassifier__max_depth': [3, 5, 8],
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.2]
}

# Define the ML Flow
with mlflow.start_run():
    # Since both precision and recall is important we use f1 score the metric
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Best estimator and details
    best_model = grid_search.best_estimator_

    # Evaluate model performance
    ypred = best_model.predict(Xtest)

    test_accuracy = accuracy_score(ytest, ypred)
    test_f1_score = f1_score(ytest, ypred)
    test_precision = precision_score(ytest, ypred)
    test_recall = recall_score(ytest, ypred)

    # Correctly log the tuning parameters as a dictionary without string description headers
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("Test Accuracy", test_accuracy)
    mlflow.log_metric("Test precision", test_precision)
    mlflow.log_metric("Test F1 Score", test_f1_score)
    mlflow.log_metric("Test Recall Score", test_recall)

    # Save the trained pipeline to deployment directory
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/artifacts/best_model.joblib")
    print("Model training completed and artifact saved successfully.")
