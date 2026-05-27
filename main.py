import os
import logging
import time

from src.utils import setup_logging
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.exploratory_analysis import generate_eda_plots
from src.model_training import train_and_select_model
from src.model_evaluation import evaluate_and_explain
from src.prediction_pipeline import run_prediction_pipeline
from src.segmentation import segment_employees
from src.business_metrics import calculate_business_metrics
from src.reporting import generate_insights_report
from src.config import CLEANED_DATA_PATH

logger = logging.getLogger("main_pipeline")

def main():
    """
    Main orchestration entrypoint for the HR Attrition Intelligence platform.
    Runs all pipeline stages sequentially and logs progress.
    """
    setup_logging()
    logger.info("==================================================================")
    logger.info("         STARTING HR ATTRITION Retention INTELLIGENCE SYSTEM      ")
    logger.info("==================================================================")
    
    start_time = time.time()
    
    try:
        # Stage 1: Data cleaning and standardization
        logger.info("[STAGE 1/8] Cleaning and validating raw dataset...")
        clean_data()
        
        # Stage 2: Feature Engineering
        logger.info("[STAGE 2/8] Running feature engineering pipeline...")
        engineer_features()
        
        # Stage 3: Exploratory Data Analysis
        logger.info("[STAGE 3/8] Generating corporate visualization plots...")
        generate_eda_plots()
        
        # Stage 4: Machine Learning Model Training
        logger.info("[STAGE 4/8] Training, tuning, and selecting champion classifier...")
        model, X_train, X_test, y_train, y_test, metrics = train_and_select_model()
        
        # Stage 5: Explainability and Evaluation Diagnostics
        logger.info("[STAGE 5/8] Running model explainability (SHAP) & evaluation...")
        evaluate_and_explain(X_test=X_test, y_test=y_test, X_train=X_train)
        
        # Stage 6: Employee Attrition Probability Risk Scoring
        logger.info("[STAGE 6/8] Generating active flight risk scores and recommendations...")
        run_prediction_pipeline()
        
        # Stage 7: Strategic Employee Segmentation
        logger.info("[STAGE 7/8] Running employee strategic segmentation...")
        segment_employees()
        
        # Stage 8: Business Costing scorecard & Executive Reporting
        logger.info("[STAGE 8/8] Calculating business metrics & writing executive report...")
        segmented_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "segmented_hr_data.csv")
        calculate_business_metrics(segmented_path)
        generate_insights_report()
        
        elapsed_time = time.time() - start_time
        logger.info("==================================================================")
        logger.info(f"   PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS   ")
        logger.info("==================================================================")
        
    except Exception as e:
        logger.critical(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
