import os
import logging
import pandas as pd
import numpy as np
from src.config import CLEANED_DATA_PATH
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def encode_travel(travel_str):
    """
    Maps BusinessTravel text categories to numerical levels.
    """
    mapping = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }
    return mapping.get(travel_str, 1) # Default to 1 (rarely) if not matched

def engineer_features(cleaned_path=CLEANED_DATA_PATH, output_path=None):
    """
    Loads cleaned data, computes the 10 corporate HR metrics, and outputs the enriched dataset.
    """
    logger.info("Starting feature engineering pipeline...")
    verify_paths_exist(cleaned_path)
    
    df = pd.read_csv(cleaned_path)
    logger.info(f"Loaded cleaned dataset with shape: {df.shape}")

    # Helper: Convert BusinessTravel to numeric for mathematical risk formulations
    travel_numeric = df["business_travel"].apply(encode_travel)

    # 1. Income Band
    # Bins MonthlyIncome into 4 corporate salary tiers
    conditions_income = [
        (df["monthly_income"] < 3000),
        (df["monthly_income"] >= 3000) & (df["monthly_income"] < 6000),
        (df["monthly_income"] >= 6000) & (df["monthly_income"] < 12000),
        (df["monthly_income"] >= 12000)
    ]
    choices_income = ["Low", "Medium", "High", "Executive"]
    df["income_band"] = np.select(conditions_income, choices_income, default="Medium")
    logger.info("Feature 1/10: 'income_band' created.")

    # 2. Tenure Group
    # Bins YearsAtCompany into cohorts
    conditions_tenure = [
        (df["years_at_company"] <= 1),
        (df["years_at_company"] > 1) & (df["years_at_company"] <= 4),
        (df["years_at_company"] > 4) & (df["years_at_company"] <= 9),
        (df["years_at_company"] > 9)
    ]
    choices_tenure = ["New Hire", "Junior", "Experienced", "Veteran"]
    df["tenure_group"] = np.select(conditions_tenure, choices_tenure, default="Experienced")
    logger.info("Feature 2/10: 'tenure_group' created.")

    # 3. Promotion Delay Risk
    # Ratio representing years since last promotion relative to tenure in current role
    df["promotion_delay_risk"] = df["years_since_last_promotion"] / (df["years_in_current_role"] + 1.0)
    logger.info("Feature 3/10: 'promotion_delay_risk' created.")

    # 4. Overtime Risk Score
    # Combines overtime status, low work-life balance, and low job involvement to flag work fatigue
    # Max value is 1 * 4 * 4 = 16, min is 0
    df["overtime_risk_score"] = df["overtime"] * (5 - df["work_life_balance"]) * (5 - df["job_involvement"])
    logger.info("Feature 4/10: 'overtime_risk_score' created.")

    # 5. Work-Life Balance Risk
    # Synthesizes low work life balance ratings, business travel frequency, and commute distance
    # Max value is 4 * (1+2) + (29/10) = 14.9
    df["work_life_balance_risk"] = (5 - df["work_life_balance"]) * (1 + travel_numeric) + (df["distance_from_home"] / 10.0)
    logger.info("Feature 5/10: 'work_life_balance_risk' created.")

    # 6. Satisfaction Composite Score
    # Simple average of 3 core satisfaction scores
    df["satisfaction_composite_score"] = (df["environment_satisfaction"] + df["job_satisfaction"] + df["relationship_satisfaction"]) / 3.0
    logger.info("Feature 6/10: 'satisfaction_composite_score' created.")

    # 7. Manager Stability Score
    # Proportion of time at company spent reporting to the current manager
    df["manager_stability_score"] = df["years_with_curr_manager"] / (df["years_at_company"] + 1.0)
    logger.info("Feature 7/10: 'manager_stability_score' created.")

    # 8. Career Growth Risk Indicator
    # Ratio representing total working years relative to current job level (lower is faster growth, higher is potential stagnation)
    df["career_growth_risk_indicator"] = df["total_working_years"] / (df["job_level"] + 1.0)
    logger.info("Feature 8/10: 'career_growth_risk_indicator' created.")

    # 9. Travel Burden Score
    # Combines business travel frequency and commute distance
    df["travel_burden_score"] = travel_numeric * df["distance_from_home"]
    logger.info("Feature 9/10: 'travel_burden_score' created.")

    # 10. Burnout Risk Index
    # Aggregate index combining overtime pressure, work-life balance risk, and low job satisfaction
    # Max value is 16 + 14.9 + 4 = 34.9
    df["burnout_risk_index"] = df["overtime_risk_score"] + df["work_life_balance_risk"] + (5 - df["job_satisfaction"])
    logger.info("Feature 10/10: 'burnout_risk_index' created.")

    # Export Enriched Dataset
    if output_path is None:
        output_path = cleaned_path  # By default overwrite cleaned file with the feature-engineered version
        
    df.to_csv(output_path, index=False)
    logger.info(f"Enriched dataset exported successfully to: {output_path} (Shape: {df.shape})")
    
    return df

if __name__ == "__main__":
    setup_logging()
    engineer_features()
