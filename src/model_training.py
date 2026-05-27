import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import (
    CLEANED_DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, HYPERPARAMETER_GRIDS
)
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Constructs a ColumnTransformer that scales numerical attributes and one-hot encodes categorical ones.
    """
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def train_and_select_model(data_path=CLEANED_DATA_PATH, model_path=MODEL_PATH):
    """
    Loads enriched data, performs stratified train-test splits, tunes multiple ML models,
    selects the champion model based on test F1-score, and saves the complete pipeline.
    """
    logger.info("Starting machine learning model training pipeline...")
    verify_paths_exist(data_path)

    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Identify Target and ID columns
    target_col = "attrition"
    id_col = "employee_id"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    # Drop non-predictive columns from feature set
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    
    # 2. Automatically Segment Column Types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    logger.info(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # Calculate balance ratio for XGBoost pos_weight scaling
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    logger.info(f"Class ratio (Negative/Positive) in train set: {ratio:.2f}")

    # 4. Prepare Models & Grids
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    models = {
        "logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', scale_pos_weight=ratio),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    best_models = {}
    model_metrics = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 5. Grid Search Tuning Loop
    for model_name, classifier in models.items():
        logger.info(f"Tuning hyper-parameters for {model_name}...")
        
        # Build composite pipeline: preprocessing + model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Fetch grid from config
        param_grid = HYPERPARAMETER_GRIDS.get(model_name, {})
        
        # Grid Search optimizing for F1-score (balances precision and recall)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        best_models[model_name] = best_pipeline
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Evaluate on Test Set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        model_metrics[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
        logger.info(
            f"[{model_name.upper()}] Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )

    # 6. Automatic Champion Selection (based on test F1-score)
    champion_name = max(model_metrics, key=lambda k: model_metrics[k]["f1_score"])
    champion_pipeline = best_models[champion_name]
    logger.info(f"*** CHAMPION MODEL SELECTED: {champion_name.upper()} (F1-score: {model_metrics[champion_name]['f1_score']:.4f}) ***")

    # 7. Save Champion Model Pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(champion_pipeline, model_path)
    logger.info(f"Serialized champion pipeline saved to: {model_path}")

    # Log metrics summary table
    metrics_df = pd.DataFrame(model_metrics).T
    logger.info("\n" + metrics_df.to_string())

    return champion_pipeline, X_train, X_test, y_train, y_test, metrics_df

if __name__ == "__main__":
    setup_logging()
    train_and_select_model()
