import os
import logging
import joblib
import pandas as pd
import numpy as np
import shap

from src.config import (
    CLEANED_DATA_PATH, EXPORT_PATH, MODEL_PATH, 
    HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD
)
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def map_retention_action(risk_level, top_factors_list):
    """
    Business rules mapping top risk factors and risk levels to actionable retention recommendations.
    """
    if risk_level == "Low":
        return "Maintain standard career engagement and annual performance reviews."
        
    factors_str = " ".join(top_factors_list).lower()
    
    # Priority rules based on risk factor strings
    if "overtime" in factors_str or "burnout" in factors_str:
        return "Implement immediate overtime cap, distribute workload, and schedule mandatory wellness recovery days."
    elif "income" in factors_str or "salary" in factors_str or "rate" in factors_str:
        return "Initiate off-cycle compensation review to align with market benchmarks; consider retention bonus."
    elif "promotion" in factors_str or "career" in factors_str:
        return "Conduct career development roadmap meeting; define clear progression milestones and promotion timeline."
    elif "satisfaction" in factors_str:
        return "Conduct structured stay-interview; evaluate department culture fit or potential lateral transfer."
    elif "manager" in factors_str:
        return "Facilitate skip-level leadership check-in; pair employee with a senior mentor outside direct line of command."
    elif "travel" in factors_str or "distance" in factors_str:
        return "Transition employee to hybrid/remote work arrangement; reduce business travel frequency by 25%."
    else:
        return "Schedule strategic check-in with HR to discuss career progression, role expectations, and role-fit."

def run_prediction_pipeline(data_path=CLEANED_DATA_PATH, model_path=MODEL_PATH, export_path=EXPORT_PATH):
    """
    Executes the risk scoring pipeline, identifies top risk drivers per employee,
    assigns retention actions, and exports scores to CSV.
    """
    logger.info("Executing prediction and risk scoring pipeline...")
    verify_paths_exist(data_path, model_path)

    # 1. Load Data and Model
    df = pd.read_csv(data_path)
    pipeline = joblib.load(model_path)
    
    # Keep ID column
    employee_ids = df["employee_id"]
    
    # Drop labels
    X = df.drop(columns=["attrition", "employee_id"])

    # 2. Predict Probabilities
    logger.info("Generating predictions...")
    probabilities = pipeline.predict_proba(X)[:, 1]

    # Assign Risk Levels
    risk_levels = []
    for prob in probabilities:
        if prob >= HIGH_RISK_THRESHOLD:
            risk_levels.append("High")
        elif prob >= MEDIUM_RISK_THRESHOLD:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")

    # 3. Extract SHAP explanations per employee to find Top Risk Factors
    logger.info("Calculating SHAP values for individual risk driver analysis...")
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [name.split("__")[-1].replace("_", " ").title() for name in feature_names]

    # Calculate SHAP values
    try:
        if classifier.__class__.__name__ in ['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
        else:
            explainer = shap.Explainer(classifier, X_transformed)
            shap_values = explainer(X_transformed)
            
        # Normalize representation
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif hasattr(shap_values, "values"):
            shap_matrix = shap_values.values
            if len(shap_matrix.shape) == 3 and shap_matrix.shape[2] == 2:
                shap_matrix = shap_matrix[:, :, 1]
        else:
            shap_matrix = shap_values
            if len(shap_matrix.shape) == 3 and shap_matrix.shape[2] == 2:
                shap_matrix = shap_matrix[:, :, 1]
    except Exception as e:
        logger.error(f"SHAP explainer failed: {e}. Falling back to default baseline explanations.")
        shap_matrix = np.zeros(X_transformed.shape)

    # 4. Extract top 3 risk factors and map recommendations
    top_factors_all = []
    retention_actions = []

    for i in range(len(df)):
        # If risk level is Low, we don't necessarily need to list positive drivers
        if risk_levels[i] == "Low":
            top_factors_all.append("None (Low Attrition Risk)")
            retention_actions.append(map_retention_action("Low", []))
            continue

        # Get features with positive contribution to attrition risk
        emp_shap = shap_matrix[i]
        pos_indices = np.where(emp_shap > 0)[0]
        
        if len(pos_indices) > 0:
            # Sort by SHAP value descending
            sorted_pos_indices = pos_indices[np.argsort(emp_shap[pos_indices])[::-1]]
            top_3_indices = sorted_pos_indices[:3]
            top_factors = [cleaned_feature_names[idx] for idx in top_3_indices]
            top_factors_str = "; ".join(top_factors)
        else:
            # Fallback if no positive SHAP values (model thinks employee is very safe)
            top_factors = ["Compensation/Role Baseline"]
            top_factors_str = "Baseline Factors"
            
        top_factors_all.append(top_factors_str)
        
        # Determine retention recommendations
        action = map_retention_action(risk_levels[i], top_factors)
        retention_actions.append(action)

    # 5. Assemble final export dataframe
    export_df = pd.DataFrame({
        "EmployeeID": employee_ids,
        "AttritionProbability": np.round(probabilities, 4),
        "RiskLevel": risk_levels,
        "TopRiskFactors": top_factors_all,
        "RecommendedRetentionAction": retention_actions
    })

    # Export to CSV
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    export_df.to_csv(export_path, index=False)
    logger.info(f"Successfully generated and exported employee risk scores to: {export_path}")
    logger.info(f"Summary of Risk Levels:\n{export_df['RiskLevel'].value_counts()}")

    return export_df

if __name__ == "__main__":
    setup_logging()
    # run_prediction_pipeline()
