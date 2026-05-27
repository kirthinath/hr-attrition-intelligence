# Streamlit Cloud Deployment Instructions

## Prerequisites
- A GitHub account (free)
- A Streamlit Community Cloud account (free) at https://share.streamlit.io

---

## Step 1: Push Your Project to GitHub

Open your terminal and run the following commands from inside your project folder:

```bash
# Initialize git (skip if already done)
git init

# Add all files
git add .

# Commit
git commit -m "feat: add streamlit dashboard"

# Create a new repo on GitHub at https://github.com/new
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/hr_attrition_intelligence.git
git branch -M main
git push -u origin main
```

> IMPORTANT: Make sure the following files are all committed and pushed:
> - streamlit_app.py
> - pages/ (all 5 page files)
> - .streamlit/config.toml
> - requirements_streamlit.txt
> - models/best_attrition_model.pkl
> - data/exports/employee_attrition_risk_scores.csv
> - data/exports/workforce_scorecard.json
> - data/processed/segmented_hr_data.csv
> - reports/figures/ (all PNG files)

---

## Step 2: Check Your .gitignore

GitHub has a 100MB file size limit. Your `.pkl` model file is small (a few MB) so it is fine.
If you have a `.gitignore` that excludes `models/` or `data/`, remove those exclusions.

---

## Step 3: Deploy to Streamlit Community Cloud

1. Go to: https://share.streamlit.io
2. Sign in with your GitHub account.
3. Click **"New app"**.
4. Fill in:
   - **Repository**: `YOUR_USERNAME/hr_attrition_intelligence`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **"Advanced settings"** and set the Python version to `3.11`.
6. Click **"Deploy!"**

Streamlit will install all packages from `requirements_streamlit.txt` automatically.
The deployment takes approximately 2-4 minutes.

---

## Step 4: Get Your Public URL

Once deployed, Streamlit will give you a public URL like:
```
https://your-username-hr-attrition-intelligence-streamlit-app-xxxxxx.streamlit.app
```

This is the URL you add to your resume and LinkedIn profile.

---

## Step 5: Test All Pages

After deployment, verify:
- [ ] Landing page loads with the 5-page navigation cards
- [ ] Executive Command Center shows KPI metrics
- [ ] Live Risk Predictor returns a prediction when you click "Run Risk Analysis"
- [ ] Workforce Risk Intelligence loads the employee table
- [ ] Attrition Drivers shows the feature importance chart
- [ ] Strategic Insights shows all 10 insights as expandable cards

---

## Troubleshooting

**"Module not found" error**: Check that all packages are in `requirements_streamlit.txt` and the versions are compatible.

**"File not found" error**: A data file or model was not pushed to GitHub. Run `git status` to check.

**Slow first load**: The first load includes importing XGBoost and loading the model. Subsequent loads use Streamlit's cache and are fast.

---

## Running Locally (for testing)

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open at http://localhost:8501
