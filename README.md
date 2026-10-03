# Graduate Outcomes Explorer — Bundle-only deployment

This folder is the minimal Streamlit Cloud deployment package. It is intentionally self-contained: `graduate_outcomes_model_bundle.joblib` holds the fitted salary and employment pipelines, clustering scaler/model/labels, dropdown mappings, forecasted programme table, cluster summary, and evaluation metrics.

## Files

```text
app.py                                  # Streamlit app entry point
graduate_outcomes_model_bundle.joblib   # Complete fitted model bundle and app tables
requirements.txt                        # Exact package versions used to create the bundle
runtime.txt                             # Python 3.12 runtime for Streamlit Cloud
```

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the **contents** of this folder — not the containing folder itself.
3. On Streamlit Community Cloud, choose **Create app**, select the repository and branch, and set the main file to `app.py`.
4. Deploy.

## Rebuilding the model bundle

The model bundle must be built in the same Python/scikit-learn environment as the training pipelines. If you retrain the models, rebuild the bundle, replace `graduate_outcomes_model_bundle.joblib`, and push the new file to GitHub.

## Interpretation

The app provides model-estimated programme-level patterns. It does not predict or guarantee an individual graduate's salary or employment outcome.
