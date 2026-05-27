import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CLEANED_DATA_PATH, FIGURES_DIR, PLOT_STYLE
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

# Apply visual configurations
plt.rcParams['font.family'] = PLOT_STYLE["font_family"]
plt.rcParams['text.color'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['axes.labelcolor'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['xtick.color'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['ytick.color'] = PLOT_STYLE["neutral_dark"]

def generate_eda_plots(data_path=CLEANED_DATA_PATH, figures_dir=FIGURES_DIR):
    """
    Generates 10 premium visualizations for EDA and saves them to reports/figures/
    """
    logger.info("Starting EDA visualization pipeline...")
    verify_paths_exist(data_path)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset for EDA with shape: {df.shape}")

    # Set style context
    sns.set_theme(style="whitegrid")
    
    # Common color palette mapping
    palette_attrition = {"Active": PLOT_STYLE["primary_color"], "Departed": PLOT_STYLE["accent_color"]}
    attrition_labels = {0: "Active", 1: "Departed"}
    df_plot = df.copy()
    df_plot["Attrition Status"] = df_plot["attrition"].map(attrition_labels)
    
    # ----------------------------------------------------
    # Plot 1: Attrition Distribution
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    counts = df_plot["Attrition Status"].value_counts()
    colors = [PLOT_STYLE["primary_color"], PLOT_STYLE["accent_color"]]
    
    # Pie chart showing proportions
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
            colors=colors, textprops={'fontsize': 11, 'weight': 'bold'})
    plt.title("Workforce Attrition Composition (Overall)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plot1_path = os.path.join(figures_dir, "01_attrition_distribution.png")
    plt.savefig(plot1_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 1: Attrition Distribution")

    # ----------------------------------------------------
    # Plot 2: Department Attrition Rate
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    dept_attrition = df.groupby("department")["attrition"].mean().reset_index()
    dept_attrition["attrition"] *= 100 # Convert to percentage
    
    ax = sns.barplot(
        data=dept_attrition, 
        x="department", 
        y="attrition", 
        palette=[PLOT_STYLE["primary_color"], PLOT_STYLE["secondary_color"], "#8FAADC"]
    )
    plt.title("Attrition Rate by Business Department", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Department", fontsize=11, labelpad=10)
    plt.ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                    ha='center', va='center', fontsize=10, weight='bold', color=PLOT_STYLE["neutral_dark"])
                    
    sns.despine()
    plt.tight_layout()
    plot2_path = os.path.join(figures_dir, "02_department_attrition.png")
    plt.savefig(plot2_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 2: Department Attrition")

    # ----------------------------------------------------
    # Plot 3: Salary vs Attrition (Income Distribution)
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_plot, x="monthly_income", hue="Attrition Status", fill=True, 
                palette=palette_attrition, common_norm=False, alpha=0.4, linewidth=2)
    plt.title("Monthly Income Distribution by Attrition Status", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Monthly Income ($)", fontsize=11, labelpad=10)
    plt.ylabel("Probability Density", fontsize=11, labelpad=10)
    sns.despine()
    plt.tight_layout()
    plot3_path = os.path.join(figures_dir, "03_salary_vs_attrition.png")
    plt.savefig(plot3_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 3: Salary vs Attrition")

    # ----------------------------------------------------
    # Plot 4: Job Role Attrition
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    role_attrition = df.groupby("job_role")["attrition"].mean().reset_index()
    role_attrition["attrition"] *= 100
    role_attrition = role_attrition.sort_values("attrition", ascending=False)
    
    sns.barplot(
        data=role_attrition, 
        y="job_role", 
        x="attrition", 
        palette=sns.color_palette("Blues_r", len(role_attrition))
    )
    plt.title("Attrition Rate by Specific Job Role", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    plt.ylabel("Job Role", fontsize=11, labelpad=10)
    sns.despine()
    plt.tight_layout()
    plot4_path = os.path.join(figures_dir, "04_job_role_attrition.png")
    plt.savefig(plot4_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 4: Job Role Attrition")

    # ----------------------------------------------------
    # Plot 5: Overtime Impact on Attrition
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    ot_attrition = df.groupby("overtime")["attrition"].mean().reset_index()
    ot_attrition["attrition"] *= 100
    ot_attrition["overtime_label"] = ot_attrition["overtime"].map({1: "Works Overtime", 0: "No Overtime"})
    
    ax = sns.barplot(
        data=ot_attrition, 
        x="overtime_label", 
        y="attrition", 
        palette=[PLOT_STYLE["primary_color"], PLOT_STYLE["accent_color"]]
    )
    plt.title("Overtime Working Status Impact on Attrition Rate", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Overtime Status", fontsize=11, labelpad=10)
    plt.ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                    ha='center', va='center', fontsize=10, weight='bold', color=PLOT_STYLE["neutral_dark"])
                    
    sns.despine()
    plt.tight_layout()
    plot5_path = os.path.join(figures_dir, "05_overtime_impact.png")
    plt.savefig(plot5_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 5: Overtime Impact")

    # ----------------------------------------------------
    # Plot 6: Satisfaction Analysis (Composite Score Distribution)
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_plot, 
        x="Attrition Status", 
        y="satisfaction_composite_score", 
        palette=palette_attrition,
        width=0.4
    )
    plt.title("Composite Job Satisfaction Score by Attrition Status", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Attrition Status", fontsize=11, labelpad=10)
    plt.ylabel("Satisfaction Composite Score (1.0 - 4.0)", fontsize=11, labelpad=10)
    sns.despine()
    plt.tight_layout()
    plot6_path = os.path.join(figures_dir, "06_satisfaction_analysis.png")
    plt.savefig(plot6_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 6: Satisfaction Analysis")

    # ----------------------------------------------------
    # Plot 7: Correlation Heatmap
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Select numerical columns and our custom engineered features
    cols_to_corr = [
        "attrition", "age", "monthly_income", "overtime", "distance_from_home", 
        "promotion_delay_risk", "overtime_risk_score", "work_life_balance_risk", 
        "satisfaction_composite_score", "manager_stability_score", "career_growth_risk_indicator", 
        "travel_burden_score", "burnout_risk_index"
    ]
    corr_df = df[cols_to_corr].corr()
    
    # Rename columns in correlation matrix for display
    display_names = {c: c.replace("_", " ").title() for c in cols_to_corr}
    corr_df = corr_df.rename(columns=display_names, index=display_names)

    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap=sns.diverging_palette(240, 10, as_cmap=True), 
        vmin=-1, 
        vmax=1, 
        square=True, 
        cbar_kws={"shrink": .8},
        annot_kws={"size": 8}
    )
    plt.title("Correlation Heatmap: Core Attributes & Engineered Risk Scores", fontsize=12, weight="bold", pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plot7_path = os.path.join(figures_dir, "07_correlation_heatmap.png")
    plt.savefig(plot7_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 7: Correlation Heatmap")

    # ----------------------------------------------------
    # Plot 8: Age vs Attrition
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_plot, x="age", hue="Attrition Status", fill=True, 
                palette=palette_attrition, common_norm=False, alpha=0.4, linewidth=2)
    plt.title("Age Distribution by Attrition Status", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Age (Years)", fontsize=11, labelpad=10)
    plt.ylabel("Probability Density", fontsize=11, labelpad=10)
    sns.despine()
    plt.tight_layout()
    plot8_path = os.path.join(figures_dir, "08_age_vs_attrition.png")
    plt.savefig(plot8_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 8: Age vs Attrition")

    # ----------------------------------------------------
    # Plot 9: Tenure Distribution
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df_plot, x="years_at_company", hue="Attrition Status", multiple="dodge", 
                 palette=palette_attrition, shrink=0.8, bins=15, alpha=0.7)
    plt.title("Tenure (Years at Company) Distribution by Attrition Status", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Years at Company", fontsize=11, labelpad=10)
    plt.ylabel("Employee Count", fontsize=11, labelpad=10)
    sns.despine()
    plt.tight_layout()
    plot9_path = os.path.join(figures_dir, "09_tenure_distribution.png")
    plt.savefig(plot9_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 9: Tenure Distribution")

    # ----------------------------------------------------
    # Plot 10: Business Travel Impact on Attrition
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    travel_attrition = df.groupby("business_travel")["attrition"].mean().reset_index()
    travel_attrition["attrition"] *= 100
    
    ax = sns.barplot(
        data=travel_attrition, 
        x="business_travel", 
        y="attrition", 
        palette=[PLOT_STYLE["secondary_color"], PLOT_STYLE["primary_color"], PLOT_STYLE["accent_color"]],
        order=["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )
    plt.title("Business Travel Frequency Impact on Attrition Rate", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Business Travel Frequency", fontsize=11, labelpad=10)
    plt.ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                    ha='center', va='center', fontsize=10, weight='bold', color=PLOT_STYLE["neutral_dark"])
                    
    sns.despine()
    plt.tight_layout()
    plot10_path = os.path.join(figures_dir, "10_business_travel_impact.png")
    plt.savefig(plot10_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot 10: Business Travel Impact")
    
    logger.info(f"All 10 EDA plots generated and saved in: {figures_dir}")

if __name__ == "__main__":
    setup_logging()
    generate_eda_plots()
