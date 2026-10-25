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

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

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
with mlflow.start_run(run_name="XGBoost_Grid_search"):
    # Since both precision and recall is important we use f1 score the metric
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and F1 scores
    results = grid_search.cv_results_
    metric_name = grid_search.scoring 

    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric(metric_name, mean_score)


    # Log parameters for the best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Evaluate model performance
    ytrain_pred = best_model.predict(Xtrain)
    ytest_pred = best_model.predict(Xtest)

    # Train Metrics
    train_precision = precision_score(ytrain, ytrain_pred)
    train_recall = recall_score(ytrain, ytrain_pred)
    train_accuracy = accuracy_score(ytrain, ytrain_pred)
    train_f1_score = f1_score(ytrain, ytrain_pred)


    # Test metrics
    test_precision = precision_score(ytest, ytest_pred)
    test_recall = recall_score(ytest, ytest_pred)
    test_accuracy = accuracy_score(ytest, ytest_pred)
    test_f1_score = f1_score(ytest, ytest_pred)

    # Log Metrics
    mlflow.log_metrics({
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1_score": train_f1_score,
        "test_f1_score": test_f1_score
    })

    # Save the trained pipeline to artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.joblib")

    logger.info("Model training completed and artifact saved successfully.")
