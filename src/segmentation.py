import os
import logging
import pandas as pd
import numpy as np
from src.config import CLEANED_DATA_PATH, EXPORT_PATH
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def segment_employees(cleaned_data_path=CLEANED_DATA_PATH, risk_scores_path=EXPORT_PATH, output_path=None):
    """
    Applies business logic to segment employees into strategic cohorts,
    merging model predictions and feature datasets.
    """
    logger.info("Executing employee strategic segmentation pipeline...")
    verify_paths_exist(cleaned_data_path, risk_scores_path)

    # 1. Load Datasets
    df_feats = pd.read_csv(cleaned_data_path)
    df_scores = pd.read_csv(risk_scores_path)

    # Merge based on employee_id / EmployeeID
    # Ensure columns match
    if "employee_id" not in df_feats.columns:
        raise ValueError("Dataframe lacks 'employee_id' column.")
    
    # Merge risk scores into features
    df = pd.merge(
        df_feats, 
        df_scores[["EmployeeID", "AttritionProbability", "RiskLevel", "TopRiskFactors"]], 
        left_on="employee_id", 
        right_on="EmployeeID", 
        how="inner"
    )
    df = df.drop(columns=["EmployeeID"]) # Clean up redundant ID column
    logger.info(f"Merged features and risk scores. Enriched shape: {df.shape}")

    # 2. Define Segment Binary Conditions
    # - Champions: High satisfaction, high job involvement, high performance, low risk
    champions_cond = (
        (df["satisfaction_composite_score"] >= 3.0) & 
        (df["job_involvement"] >= 3) & 
        (df["performance_rating"] >= 3) & 
        (df["RiskLevel"] == "Low")
    )

    # - Loyal Employees: Long tenure, stable manager, low/med risk
    loyal_cond = (
        (df["years_at_company"] >= 5) & 
        (df["manager_stability_score"] >= 0.6) & 
        (df["RiskLevel"].isin(["Low", "Medium"]))
    )

    # - High Performers at Risk: High performance rating, high risk level
    high_perf_at_risk_cond = (
        (df["performance_rating"] >= 3) & 
        (df["RiskLevel"] == "High")
    )

    # - Burnout Risk Employees: High overtime risk or high burnout risk index
    burnout_cond = (
        (df["overtime_risk_score"] >= 4) | 
        (df["burnout_risk_index"] >= 10)
    )

    # - Disengaged Employees: Low job involvement and low job satisfaction
    disengaged_cond = (
        (df["job_involvement"] <= 2) & 
        (df["job_satisfaction"] <= 2)
    )

    # - New Employees: Short tenure
    new_employee_cond = (df["years_at_company"] <= 2)

    # - Flight Risk Employees: High risk level, low satisfaction
    flight_risk_cond = (
        (df["RiskLevel"] == "High") & 
        (df["satisfaction_composite_score"] < 2.5)
    )

    # Assign Binary Segment Columns
    df["segment_champions"] = champions_cond.astype(int)
    df["segment_loyal"] = loyal_cond.astype(int)
    df["segment_high_performers_at_risk"] = high_perf_at_risk_cond.astype(int)
    df["segment_burnout_risk"] = burnout_cond.astype(int)
    df["segment_disengaged"] = disengaged_cond.astype(int)
    df["segment_new_employees"] = new_employee_cond.astype(int)
    df["segment_flight_risk"] = flight_risk_cond.astype(int)

    # 3. Determine Prioritized Primary Segment
    # Priority order:
    # 1. High Performers at Risk
    # 2. Flight Risk
    # 3. Burnout Risk
    # 4. Disengaged
    # 5. New Employee
    # 6. Champion
    # 7. Loyal
    
    primary_segment = []
    for idx, row in df.iterrows():
        if row["segment_high_performers_at_risk"] == 1:
            primary_segment.append("High Performer at Risk")
        elif row["segment_flight_risk"] == 1:
            primary_segment.append("Flight Risk")
        elif row["segment_burnout_risk"] == 1:
            primary_segment.append("Burnout Risk")
        elif row["segment_disengaged"] == 1:
            primary_segment.append("Disengaged")
        elif row["segment_new_employees"] == 1:
            primary_segment.append("New Employee")
        elif row["segment_champions"] == 1:
            primary_segment.append("Champion")
        elif row["segment_loyal"] == 1:
            primary_segment.append("Loyal Employee")
        else:
            primary_segment.append("General Workforce")

    df["primary_segment"] = primary_segment
    logger.info("Workforce segmentation complete.")

    # Log Segment distribution
    distribution = df["primary_segment"].value_counts()
    logger.info(f"Segment Distribution:\n{distribution.to_string()}")

    # 4. Save Output
    if output_path is None:
        # Save to data/processed/segmented_hr_data.csv
        output_path = os.path.join(os.path.dirname(cleaned_data_path), "segmented_hr_data.csv")
    
    df.to_csv(output_path, index=False)
    logger.info(f"Segmented dataset exported successfully to: {output_path}")

    return df

if __name__ == "__main__":
    setup_logging()
    # segment_employees()
