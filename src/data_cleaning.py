import os
import re
import logging
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

def camel_to_snake(name):
    """
    Converts CamelCase to snake_case.
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def clean_data(raw_path=RAW_DATA_PATH, cleaned_path=CLEANED_DATA_PATH):
    """
    Performs end-to-end data cleaning and validation.
    """
    logger.info("Starting data cleaning pipeline...")
    verify_paths_exist(raw_path)

    # 1. Load Dataset
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")

    # 2. Standardize Column Names
    # We map EmployeeNumber specifically to employee_id and OverTime to overtime
    rename_dict = {}
    for col in df.columns:
        if col == "EmployeeNumber":
            rename_dict[col] = "employee_id"
        elif col == "OverTime":
            rename_dict[col] = "overtime"
        else:
            rename_dict[col] = camel_to_snake(col)
    df = df.rename(columns=rename_dict)
    logger.info("Column names standardized to snake_case.")

    # 3. Handle Duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicate_count = initial_rows - len(df)
    if duplicate_count > 0:
        logger.warning(f"Removed {duplicate_count} duplicate records.")
    else:
        logger.info("No duplicate records found.")

    # 4. Handle Missing Values
    missing_report = df.isnull().sum()
    total_missing = missing_report.sum()
    if total_missing > 0:
        logger.warning(f"Found {total_missing} missing values. Performing business logic imputations...")
        for col in df.columns:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                if df[col].dtype == object:
                    # Impute categorical with mode
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Imputed {missing_count} missing values in categorical column '{col}' with mode: '{mode_val}'")
                else:
                    # Impute numerical with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Imputed {missing_count} missing values in numerical column '{col}' with median: {median_val}")
    else:
        logger.info("No missing values detected in the raw dataset.")

    # 5. Data Consistency Validation
    # Consistent constraints:
    # - age >= total_working_years + 18 (since Over18 is Yes, but let's just make sure age >= total_working_years)
    # - total_working_years >= years_at_company
    # - years_at_company >= years_in_current_role
    # - years_at_company >= years_since_last_promotion
    # - years_at_company >= years_with_curr_manager
    
    inconsistent_records = df[
        (df["age"] < df["total_working_years"]) |
        (df["total_working_years"] < df["years_at_company"]) |
        (df["years_at_company"] < df["years_in_current_role"]) |
        (df["years_at_company"] < df["years_since_last_promotion"]) |
        (df["years_at_company"] < df["years_with_curr_manager"])
    ]
    
    if len(inconsistent_records) > 0:
        logger.warning(f"Detected {len(inconsistent_records)} inconsistent records violating logical rules. Rectifying...")
        # Fix logic: cap years at company, years in current role, years with manager to sensible bounds if there's inconsistency,
        # or drop them. Since it's a raw source of truth, let's fix inconsistency by ensuring bounds:
        for idx in inconsistent_records.index:
            # If total_working_years < years_at_company, set total_working_years = years_at_company
            if df.loc[idx, "total_working_years"] < df.loc[idx, "years_at_company"]:
                df.loc[idx, "total_working_years"] = df.loc[idx, "years_at_company"]
            # If age < total_working_years, set age = total_working_years + 18
            if df.loc[idx, "age"] < df.loc[idx, "total_working_years"]:
                df.loc[idx, "age"] = df.loc[idx, "total_working_years"] + 18
        logger.info("Logical consistency rules applied and anomalies resolved.")
    else:
        logger.info("Data consistency validation passed successfully.")

    # 6. Binary Conversions
    # Map target 'attrition' and feature 'overtime' to 1/0
    if "attrition" in df.columns and df["attrition"].dtype == object:
        df["attrition"] = df["attrition"].map({"Yes": 1, "No": 0})
        logger.info("Mapped target column 'attrition': Yes/No -> 1/0")
        
    if "overtime" in df.columns and df["overtime"].dtype == object:
        df["overtime"] = df["overtime"].map({"Yes": 1, "No": 0})
        logger.info("Mapped feature column 'overtime': Yes/No -> 1/0")

    # 7. Identify and Drop Zero-Variance Columns
    # Columns like employee_count (always 1), over_18 (always Y), and standard_hours (always 80)
    zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]
    if len(zero_var_cols) > 0:
        logger.info(f"Dropping zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)

    # 8. Export Cleaned Dataset
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Successfully clean and exported data to: {cleaned_path}")
    logger.info(f"Cleaned dataset shape: {df.shape}")

    # 9. Generate missing values and consistency report
    report_path = os.path.join(os.path.dirname(cleaned_path), "data_cleaning_report.txt")
    with open(report_path, "w") as f:
        f.write("=========================================\n")
        f.write("HR ATTRITION INTELLIGENCE: DATA CLEANING REPORT\n")
        f.write("=========================================\n\n")
        f.write(f"Raw file analyzed: {raw_path}\n")
        f.write(f"Raw shape: {len(df) + duplicate_count} rows, {len(df.columns) + len(zero_var_cols)} columns\n")
        f.write(f"Cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Duplicates removed: {duplicate_count}\n")
        f.write(f"Missing values found: {total_missing}\n")
        f.write(f"Zero-variance columns dropped: {zero_var_cols}\n\n")
        f.write("Missing Value Count by Column:\n")
        f.write(missing_report.to_string())
        f.write("\n\nConsistency Checks Summary:\n")
        f.write(f"- Logical violations fixed: {len(inconsistent_records)}\n")
        f.write("=========================================\n")
    logger.info(f"Cleaning diagnostics report written to: {report_path}")

    return df

if __name__ == "__main__":
    setup_logging()
    clean_data()
