import os

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "employee_attrition_raw.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_hr_data.csv")
EXPORT_DIR = os.path.join(BASE_DIR, "data", "exports")
EXPORT_PATH = os.path.join(EXPORT_DIR, "employee_attrition_risk_scores.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_attrition_model.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
INSIGHTS_REPORT_PATH = os.path.join(REPORTS_DIR, "insights_report.md")
SQL_DIR = os.path.join(BASE_DIR, "sql")

# Ensure critical directories exist
for path in [os.path.dirname(CLEANED_DATA_PATH), EXPORT_DIR, MODEL_DIR, FIGURES_DIR, REPORTS_DIR, SQL_DIR]:
    os.makedirs(path, exist_ok=True)

# Visual Style Settings (Corporate Palette)
PLOT_STYLE = {
    "font_family": "DejaVu Sans",
    "primary_color": "#1F4E79",       # Deep Navy
    "secondary_color": "#2E75B6",     # Muted Blue
    "accent_color": "#C55A11",        # Amber/Rust for High Risk
    "neutral_dark": "#262626",        # Charcoal for Text
    "neutral_light": "#F2F2F2",       # Light Gray for Backgrounds
    "success_color": "#548235",       # Muted Green
    "warning_color": "#FFC000",       # Muted Gold
    "grid_color": "#D9D9D9",
    "dpi": 300
}

# Machine Learning Settings
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Business Risk Classification Thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

# HR Cost Assumptions (Industry Standard: Attrition cost is roughly 1.5x annual salary)
ATTRITION_COST_MULTIPLIER = 1.5

# Hyperparameter Tuning Grids for GridSearchCV
HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__max_iter": [1000]
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [6, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": ["balanced"]
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__subsample": [0.8, 1.0]
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__subsample": [0.8, 1.0]
    }
}
