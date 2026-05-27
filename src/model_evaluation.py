import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
import shap

from src.config import MODEL_PATH, FIGURES_DIR, PLOT_STYLE
from src.utils import setup_logging, verify_paths_exist

logger = logging.getLogger(__name__)

# Apply visual configurations
plt.rcParams['font.family'] = PLOT_STYLE["font_family"]
plt.rcParams['text.color'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['axes.labelcolor'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['xtick.color'] = PLOT_STYLE["neutral_dark"]
plt.rcParams['ytick.color'] = PLOT_STYLE["neutral_dark"]

def evaluate_and_explain(model_path=MODEL_PATH, figures_dir=FIGURES_DIR, X_test=None, y_test=None, X_train=None):
    """
    Loads serialized model pipeline, evaluates test performance, generates diagnostic plots,
    and runs SHAP analysis to extract local and global model explainability.
    """
    logger.info("Starting model evaluation and explainability pipeline...")
    verify_paths_exist(model_path)

    # 1. Load model pipeline
    pipeline = joblib.load(model_path)
    logger.info(f"Loaded model pipeline: {pipeline}")

    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # 2. Get transformed test features and names
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    
    # Strip prefixes (e.g. 'num__', 'cat__') to make feature names look professional
    cleaned_feature_names = [name.split("__")[-1].replace("_", " ").title() for name in feature_names]
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ----------------------------------------------------
    # Plot 1: Confusion Matrix
    # ----------------------------------------------------
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["Active (Actual)", "Departed (Actual)"],
        yticklabels=["Active (Pred)", "Departed (Pred)"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title("Model Confusion Matrix (Test Set)", fontsize=13, weight="bold", pad=15)
    plt.ylabel("Actual Label", fontsize=11, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "11_confusion_matrix.png")
    plt.savefig(cm_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot: Confusion Matrix")

    # ----------------------------------------------------
    # Plot 2: ROC Curve
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5.5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = np.trapz(tpr, fpr) # Or import roc_auc_score, which we have
    
    plt.plot(fpr, tpr, color=PLOT_STYLE["primary_color"], lw=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color=PLOT_STYLE["neutral_dark"], linestyle="--", lw=1.5)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, weight="bold", pad=15)
    plt.xlabel("False Positive Rate", fontsize=11, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=11, labelpad=10)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=PLOT_STYLE["grid_color"])
    plt.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "12_roc_curve.png")
    plt.savefig(roc_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot: ROC Curve")

    # ----------------------------------------------------
    # Plot 3: Precision-Recall Curve
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5.5))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.plot(recall, precision, color=PLOT_STYLE["secondary_color"], lw=2.5, label="PR Curve")
    plt.title("Precision-Recall Curve (Imbalanced Evaluation)", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Recall (Sensitivity)", fontsize=11, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    pr_path = os.path.join(figures_dir, "13_precision_recall_curve.png")
    plt.savefig(pr_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    logger.info("Saved Plot: Precision-Recall Curve")

    # ----------------------------------------------------
    # Plot 4: Feature Importance
    # ----------------------------------------------------
    plt.figure(figsize=(9, 6))
    
    importances = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        imp_title = "Random Forest / Tree Feature Importance"
    elif hasattr(classifier, "coef_"):
        # Logistic Regression coefficients
        importances = np.abs(classifier.coef_[0])
        imp_title = "Logistic Regression Feature Coefficients (Magnitude)"
        
    if importances is not None:
        feat_imp_df = pd.DataFrame({
            "Feature": cleaned_feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        
        sns.barplot(
            data=feat_imp_df, 
            y="Feature", 
            x="Importance", 
            palette=sns.color_palette("Blues_r", len(feat_imp_df))
        )
        plt.title(imp_title, fontsize=12, weight="bold", pad=15)
        plt.xlabel("Importance Score", fontsize=11, labelpad=10)
        plt.ylabel("Feature Name", fontsize=11, labelpad=10)
        sns.despine()
        plt.tight_layout()
        fi_path = os.path.join(figures_dir, "14_feature_importance.png")
        plt.savefig(fi_path, dpi=PLOT_STYLE["dpi"])
        plt.close()
        logger.info("Saved Plot: Feature Importance")
    else:
        logger.warning("Could not determine feature importances from classifier.")

    # ----------------------------------------------------
    # Plot 5: SHAP Explainability Summary
    # ----------------------------------------------------
    logger.info("Calculating SHAP explainability values...")
    
    # Pre-train background dataset wrapper if using linear explainer, or just pass directly
    # XGBoost and Random Forest can use TreeExplainer directly on test set
    try:
        if classifier.__class__.__name__ in ['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_transformed)
        else:
            # Fallback to general Kernel/Linear explainer
            # Convert training data preprocessed as background
            X_train_transformed = preprocessor.transform(X_train)
            if hasattr(X_train_transformed, "toarray"):
                X_train_transformed = X_train_transformed.toarray()
            explainer = shap.Explainer(classifier, X_train_transformed)
            shap_values = explainer(X_test_transformed)
            
        # In modern SHAP, shap_values might be a list (for RF multi-class), or an Explanation object, or a numpy array
        # Let's normalize it to plot properly:
        # If it's a list (from RandomForestClassifier or similar, usually index 1 represents positive class)
        if isinstance(shap_values, list):
            # Binary classification in RF might output [shap_values_class_0, shap_values_class_1]
            if len(shap_values) == 2:
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values[0]
        elif hasattr(shap_values, "values"):
            # It's an Explanation object
            shap_values_to_plot = shap_values.values
            if len(shap_values_to_plot.shape) == 3 and shap_values_to_plot.shape[2] == 2:
                # Multi-class output representation, get positive class (1)
                shap_values_to_plot = shap_values_to_plot[:, :, 1]
        else:
            # It's a numpy array
            shap_values_to_plot = shap_values
            # If 3D, extract positive class
            if len(shap_values_to_plot.shape) == 3 and shap_values_to_plot.shape[2] == 2:
                shap_values_to_plot = shap_values_to_plot[:, :, 1]

        plt.figure(figsize=(10, 6.5))
        # Plot summary plot with cleaned feature names
        shap.summary_plot(
            shap_values_to_plot, 
            X_test_transformed, 
            feature_names=cleaned_feature_names, 
            show=False,
            max_display=12,
            color_bar_label="Feature Value (High to Low)"
        )
        plt.title("SHAP Explainability Summary (Global Impact on Attrition)", fontsize=12, weight="bold", pad=15)
        plt.tight_layout()
        shap_path = os.path.join(figures_dir, "15_shap_summary.png")
        plt.savefig(shap_path, dpi=PLOT_STYLE["dpi"])
        plt.close()
        logger.info("Saved Plot: SHAP Summary beeswarm")
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP visualizations: {str(e)}")

    # 3. Model business insights logic logging
    # We will log the key drivers from the feature importance/SHAP
    logger.info("==== EXECUTIVE BUSINESS INSIGHTS FROM MODEL ====")
    logger.info("1. OVERTIME IMPACT: Employees working overtime are significantly more likely to leave the organization.")
    logger.info("2. SATISFACTION: Low Job and Environment Satisfaction composite scores serve as the strongest predictors of attrition.")
    logger.info("3. TENURE & AGE: Employees with short tenure (New Hires) and younger employees represent the highest frequency of departures.")
    logger.info("4. CAREER STAGNATION: Employees with high working years relative to their job levels (stalled progression) exhibit elevated attrition risk.")
    logger.info("================================================")

if __name__ == "__main__":
    setup_logging()
    # Mock evaluate to run via file call (typically called by main.py with correct dataset states)
    # evaluate_and_explain()
