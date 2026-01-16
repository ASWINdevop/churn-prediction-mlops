import pandas as pd
import shap
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')
REPORT_PATH = os.path.join(BASE_DIR, 'reports')

# Create reports directory if it doesn't exist
os.makedirs(REPORT_PATH, exist_ok=True)

def explain_model():
    """
    Generates SHAP explanations for the trained XGBoost model.
    Saves SHAP summary plot to the reports directory.
    """

    print("Loading training data and trained model...")
    X_train = pd.read_csv(TRAIN_PATH).drop(["churn", "customerid"], axis=1)

    # Load the full pipeine

    pipeline = joblib.load(MODEL_PATH)

    # Extract the model from the pipeline
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # Transform the training data
    # Transforming categorical features using the preprocessor
    print("Transforming training data for SHAP ...")
    X_transformed = preprocessor.transform(X_train)

    # Get categorical feature names back (
    # OneHotEncoder destroys them, so we rebuild them)

    cat_features = (preprocessor.named_transformers_['cat']
                     .get_feature_names_out().tolist())
     
    # If we had numeric features passed through, we'd add them here. 
    # Since we used 'passthrough' for numbers, they are at the end.

    num_features = [col for col in X_train.columns if col not in 
                        preprocessor.named_transformers_['cat'].feature_names_in_]

    all_feature_names = cat_features + num_features

    # convert to DataFrame for SHAP
    X_transformed_df = pd.DataFrame(X_transformed, columns = all_feature_names)

    # Calculate SHAP values
    # Using TreeExplainer as it is optimized for XGBoost/RandomForest model
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_transformed_df)


    # ----------------------------------------------------------------
    # PLOT1:  SHAP summary plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_explanation, X_transformed_df, show=False)

    save_path_sum = os.path.join(REPORT_PATH, 'shap_summary_plot.png')
    plt.savefig(save_path_sum, bbox_inches='tight')
    print(f"SHAP summary plot saved to {save_path_sum}")
    plt.close()
    # ----------------------------------------------------------------


    # ----------------------------------------------------------------
    #PLOT 2 : Waterfall plot
    print("Generating SHAP waterfall plot for riskiest customer...")

    # Identify the riskiest customer (highest predicted probability)
    risk_scores = shap_explanation.values.sum(axis=1) 
    highest_risk_idx = risk_scores.argmax()

    print(f"Found Customer Index:{highest_risk_idx} (Highest Risk Score)")

    # Slice the explanation for that customer
    single_customer_explanation = shap_explanation[highest_risk_idx]
    plt.figure()
    shap.plots.waterfall(single_customer_explanation, show=False, max_display=12)
    save_path_water  = os.path.join(REPORT_PATH, 'shap_waterfall_plot.png')
    plt.gcf().set_size_inches(8, 8)
    plt.gcf().savefig(save_path_water, bbox_inches='tight')
    print(f"SHAP waterfall plot saved to {save_path_water}")
    plt.close()
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # PLOT 3: Absolute Mean Bar Plot 
 
    print("🎨 Generating 2/4: Absolute Mean Bar Plot...")
    plt.figure()
    # plot_type='bar' calculates the mean absolute value for each feature
    shap.summary_plot(shap_explanation, X_transformed_df, plot_type="bar", show=False)
    plt.savefig(os.path.join(REPORT_PATH, "shap_feature_importance_bar.png"), bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------


    #  ---------------------------------------------------------------- 
    # PLOT 4: Stacked Force Plot (Interactive HTML)
    
    # Note: Force plots are interactive JavaScript. They don't save well as PNG.
    # We save this one as an HTML file you can open in Chrome/Edge.
    print("🎨 Generating 4/4: Stacked Force Plot (HTML)...")
    
    # We take a sample of 1000 customers because plotting 5000+ is slow in JS
    sample_idx = np.random.choice(X_transformed_df.shape[0], 1000, replace=False)
    
    force_plot = shap.plots.force(
        shap_explanation.base_values[sample_idx], 
        shap_explanation.values[sample_idx], 
        X_transformed_df.iloc[sample_idx],
        show=False
    )
    
    # Save as HTML
    shap.save_html(os.path.join(REPORT_PATH, "shap_force_plot_stacked.html"), force_plot)
    print(f"✅ All reports saved to {REPORT_PATH}")
    # ----------------------------------------------------------------

if __name__ == "__main__":
    explain_model()