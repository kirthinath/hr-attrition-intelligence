import os
import json
import logging
import pandas as pd
import numpy as np
from src.config import ATTRITION_COST_MULTIPLIER, EXPORT_DIR
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def calculate_business_metrics(segmented_data_path):
    """
    Computes HR workforce metrics and financial impact figures.
    """
    logger.info("Computing business metrics scorecard...")
    verify_paths_exist(segmented_data_path)

    df = pd.read_csv(segmented_data_path)

    # 1. Attrition and Retention Rates (Based on historical labels)
    total_headcount = len(df)
    departed_count = df["attrition"].sum()
    active_count = total_headcount - departed_count
    
    attrition_rate = float(departed_count) / total_headcount
    retention_rate = 1.0 - attrition_rate

    # 2. Departmental Attrition
    dept_attrition = df.groupby("department")["attrition"].mean().to_dict()

    # 3. Attrition Cost Estimate
    # Formula: 1.5x annual salary for each departed employee
    departed_df = df[df["attrition"] == 1]
    estimated_attrition_cost = np.sum(departed_df["monthly_income"] * 12 * ATTRITION_COST_MULTIPLIER)

    # 4. Employee Lifetime Value (ELV) proxy for active workforce
    # Formula: monthly_income * 12 * years_at_company * (performance_rating / 3.0)
    df["elv"] = df["monthly_income"] * 12 * df["years_at_company"] * (df["performance_rating"] / 3.0)
    avg_elv_active = df[df["attrition"] == 0]["elv"].mean()
    avg_elv_departed = df[df["attrition"] == 1]["elv"].mean()
    total_elv_loss = df[df["attrition"] == 1]["elv"].sum()

    # 5. Active High Risk Count (predictive)
    active_df = df[df["attrition"] == 0]
    high_risk_active_count = int((active_df["RiskLevel"] == "High").sum())
    medium_risk_active_count = int((active_df["RiskLevel"] == "Medium").sum())

    # 6. Overtime Impact
    ot_df = df.groupby("overtime")["attrition"].mean()
    ot_attrition_rate = ot_df.get(1, 0.0)
    no_ot_attrition_rate = ot_df.get(0, 0.0)
    ot_multiplier = ot_attrition_rate / no_ot_attrition_rate if no_ot_attrition_rate > 0 else 1.0

    # 7. Promotion Delay Impact
    # Compare attrition rate of employees with high promotion delay (years_since_last_promotion >= 3)
    promo_delay_high_df = df[df["years_since_last_promotion"] >= 3]
    promo_delay_low_df = df[df["years_since_last_promotion"] < 3]
    
    high_promo_delay_attrition = promo_delay_high_df["attrition"].mean()
    low_promo_delay_attrition = promo_delay_low_df["attrition"].mean()
    promo_delay_multiplier = high_promo_delay_attrition / low_promo_delay_attrition if low_promo_delay_attrition > 0 else 1.0

    # 8. Workforce Risk Score (Aggregate risk score of active employees)
    workforce_risk_score = active_df["AttritionProbability"].mean()

    # Assemble Scorecard
    scorecard = {
        "workforce_kpis": {
            "total_historical_headcount": int(total_headcount),
            "current_active_headcount": int(active_count),
            "historical_departed_count": int(departed_count),
            "overall_attrition_rate": round(float(attrition_rate), 4),
            "overall_retention_rate": round(float(retention_rate), 4)
        },
        "financial_impact": {
            "total_estimated_attrition_cost": round(float(estimated_attrition_cost), 2),
            "average_cost_per_departed_employee": round(float(estimated_attrition_cost / departed_count), 2) if departed_count > 0 else 0.0,
            "total_lifetime_value_loss_elv": round(float(total_elv_loss), 2),
            "average_active_employee_elv": round(float(avg_elv_active), 2),
            "average_departed_employee_elv": round(float(avg_elv_departed), 2)
        },
        "predictive_risk_exposure": {
            "active_high_risk_headcount": int(high_risk_active_count),
            "active_medium_risk_headcount": int(medium_risk_active_count),
            "aggregate_workforce_risk_score": round(float(workforce_risk_score), 4)
        },
        "overtime_vulnerability": {
            "overtime_attrition_rate": round(float(ot_attrition_rate), 4),
            "no_overtime_attrition_rate": round(float(no_ot_attrition_rate), 4),
            "overtime_attrition_multiplier": round(float(ot_multiplier), 2)
        },
        "career_progression_vulnerability": {
            "high_promotion_delay_attrition_rate": round(float(high_promo_delay_attrition), 4),
            "low_promotion_delay_attrition_rate": round(float(low_promo_delay_attrition), 4),
            "promotion_delay_attrition_multiplier": round(float(promo_delay_multiplier), 2)
        },
        "departmental_attrition_rates": {dept: round(float(rate), 4) for dept, rate in dept_attrition.items()}
    }

    # Print Scorecard
    logger.info("==================================================================")
    logger.info("                  ENTERPRISE WORKFORCE SCORECARD                 ")
    logger.info("==================================================================")
    logger.info(f"Total Headcount Analyzed      : {total_headcount}")
    logger.info(f"Overall Attrition Rate        : {attrition_rate*100:.2f}% (Retention: {retention_rate*100:.2f}%)")
    logger.info(f"Estimated Attrition Cost Loss : ${estimated_attrition_cost:,.2f}")
    logger.info(f"Active High Risk Headcount    : {high_risk_active_count} employees")
    logger.info(f"Aggregate Workforce Risk Score: {workforce_risk_score*100:.2f}%")
    logger.info(f"Overtime Attrition Multiplier : {ot_multiplier:.2f}x higher risk")
    logger.info(f"Promo Delay Attrition Mult    : {promo_delay_multiplier:.2f}x higher risk")
    logger.info("--------------------- Department Attrition ----------------------")
    for dept, rate in dept_attrition.items():
        logger.info(f"  * {dept:<28}: {rate*100:.2f}%")
    logger.info("==================================================================")

    # Export Scorecard to JSON
    scorecard_path = os.path.join(EXPORT_DIR, "workforce_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump(scorecard, f, indent=4)
    logger.info(f"Workforce scorecard exported to: {scorecard_path}")

    return scorecard

if __name__ == "__main__":
    setup_logging()
    # calculate_business_metrics()
