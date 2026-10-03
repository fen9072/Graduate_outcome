"""Graduate Outcomes Explorer — bundle-only Streamlit app.

Deployment requirement: place this file beside graduate_outcomes_model_bundle.joblib.
No source CSV, separate model file, or separate predictions CSV is required at runtime.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Graduate Outcomes Explorer",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "graduate_outcomes_model_bundle.joblib"

PROFILE_ORDER = [
    "Moderate salary, lower employment profile",
    "Moderate salary, higher employment profile",
    "Higher salary, higher employment profile",
]
PROFILE_COLOURS = {
    "Moderate salary, lower employment profile": "#E15759",
    "Moderate salary, higher employment profile": "#4E79A7",
    "Higher salary, higher employment profile": "#59A14F",
}


@st.cache_resource
def load_model_bundle() -> dict:
    """Load the one deployment file containing models, tables, and dropdown options."""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "graduate_outcomes_model_bundle.joblib was not found beside app.py."
        )

    bundle = joblib.load(BUNDLE_PATH)
    required_keys = {
        "salary_pipeline",
        "employment_pipeline",
        "cluster_scaler",
        "kmeans_model",
        "cluster_labels",
        "universities",
        "schools_by_university",
        "degrees_by_university_school",
        "programme_predictions",
        "cluster_summary",
        "evaluation_metrics",
        "prediction_input_year",
        "prediction_outcome_year",
    }
    missing = sorted(required_keys - set(bundle))
    if missing:
        raise KeyError(
            "The Joblib bundle is missing these required keys: " + ", ".join(missing)
        )
    return bundle


def programme_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, and sort programme fields for display and download."""
    fields = [
        "cluster",
        "outcome_profile_2026",
        "university",
        "school",
        "degree",
        "predicted_2026_gross_median",
        "predicted_2026_employment_rate",
    ]
    output = frame[[field for field in fields if field in frame.columns]].copy()
    return output.rename(
        columns={
            "cluster": "Cluster",
            "outcome_profile_2026": "Predicted 2026 Outcome Profile",
            "university": "University",
            "school": "School",
            "degree": "Degree",
            "predicted_2026_gross_median": "Predicted 2026 Median Salary (S$)",
            "predicted_2026_employment_rate": "Predicted 2026 Employment Rate (%)",
        }
    )


def cluster_profile(
    salary: float,
    employment_rate: float,
    bundle: dict,
) -> tuple[int, str]:
    """Classify a new pair of predictions with the already-fitted 2026 K-means model."""
    cluster_input = pd.DataFrame(
        [{
            "predicted_2026_gross_median": salary,
            "predicted_2026_employment_rate": employment_rate,
        }]
    )
    scaled_input = bundle["cluster_scaler"].transform(cluster_input)
    cluster = int(bundle["kmeans_model"].predict(scaled_input)[0])
    profile = bundle["cluster_labels"].get(cluster, f"Cluster {cluster}")
    return cluster, profile


def make_scatter(predictions: pd.DataFrame):
    """Visualise the model-estimated 2026 programme profiles."""
    figure = px.scatter(
        predictions,
        x="predicted_2026_gross_median",
        y="predicted_2026_employment_rate",
        color="outcome_profile_2026",
        category_orders={"outcome_profile_2026": PROFILE_ORDER},
        color_discrete_map=PROFILE_COLOURS,
        hover_data={
            "university": True,
            "school": True,
            "degree": True,
            "cluster": True,
            "predicted_2026_gross_median": ":,.0f",
            "predicted_2026_employment_rate": ":.1f",
            "outcome_profile_2026": False,
        },
        labels={
            "predicted_2026_gross_median": "Predicted 2026 Gross Monthly Median Salary (S$)",
            "predicted_2026_employment_rate": "Predicted 2026 Employment Rate (%)",
            "outcome_profile_2026": "Predicted outcome profile",
        },
        title="Predicted 2026 Graduate Programme Outcome Profiles",
        template="plotly_white",
        height=650,
    )
    figure.update_traces(
        marker={
            "size": 10,
            "opacity": 0.82,
            "line": {"width": 0.7, "color": "white"},
        }
    )
    figure.add_vline(
        x=predictions["predicted_2026_gross_median"].mean(),
        line_dash="dash",
        line_color="#6B7280",
        annotation_text="Overall salary average",
        annotation_position="top",
    )
    figure.add_hline(
        y=predictions["predicted_2026_employment_rate"].mean(),
        line_dash="dash",
        line_color="#6B7280",
        annotation_text="Overall employment-rate average",
        annotation_position="right",
    )
    figure.update_layout(
        title_x=0.5,
        legend_title_text="2026 Outcome Profile",
    )
    return figure


def app() -> None:
    try:
        bundle = load_model_bundle()
    except Exception as error:
        st.error(f"Unable to load the model bundle: {error}")
        st.stop()

    predictions = bundle["programme_predictions"].copy()
    cluster_summary = bundle["cluster_summary"].copy()
    evaluation_metrics = bundle["evaluation_metrics"].copy()

    st.title("Graduate Outcomes Explorer")
    st.caption(bundle.get("model_notes", "Programme-level model estimates."))

    page = st.sidebar.radio(
        "Navigate",
        [
            "2026 Predictor",
            "Outcome Profile Explorer",
            "Programme Tables",
            "Model Evaluation & Limitations",
        ],
    )

    if page == "2026 Predictor":
        st.header("2026 Programme Outcome Predictor")
        st.write(
            "Choose a programme from the 2025 records. The app estimates its 2026 "
            "gross monthly median salary, full-time permanent employment rate, and outcome profile."
        )

        university = st.selectbox("University", bundle["universities"])
        school = st.selectbox(
            "School",
            bundle["schools_by_university"][university],
        )
        degree = st.selectbox(
            "Degree",
            bundle["degrees_by_university_school"][(university, school)],
        )

        if st.button("Estimate 2026 outcomes", type="primary"):
            record = pd.DataFrame([{
                "year": bundle["prediction_input_year"],
                "university": university,
                "school": school,
                "degree": degree,
            }])

            predicted_salary = float(
                bundle["salary_pipeline"].predict(record)[0]
            )
            predicted_employment = float(
                bundle["employment_pipeline"].predict(record)[0]
            )
            cluster, profile = cluster_profile(
                predicted_salary,
                predicted_employment,
                bundle,
            )

            col_salary, col_employment, col_profile = st.columns(3)
            col_salary.metric(
                "Predicted 2026 median salary",
                f"S${predicted_salary:,.0f}",
            )
            col_employment.metric(
                "Predicted 2026 employment rate",
                f"{predicted_employment:.1f}%",
            )
            col_profile.metric("Predicted profile", f"Cluster {cluster}")
            st.success(profile)
            st.info(
                "These are model-estimated programme-level patterns. They do not predict "
                "or guarantee an individual graduate's salary or employment outcome."
            )

    elif page == "Outcome Profile Explorer":
        st.header("Predicted 2026 Outcome Profiles")
        st.write(
            "Each point represents one university programme. Profiles are formed by K-means "
            "clustering of model-predicted salary and employment outcomes."
        )
        st.plotly_chart(make_scatter(predictions), width="stretch")

        st.subheader("Cluster summary")
        display_summary = cluster_summary.rename(
            columns={
                "cluster": "Cluster",
                "outcome_profile_2026": "Predicted 2026 Outcome Profile",
                "programmes": "Programmes",
                "mean_gross_median_salary": "Mean Predicted Salary (S$)",
                "mean_employment_rate": "Mean Predicted Employment Rate (%)",
            }
        )
        st.dataframe(
            display_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Mean Predicted Salary (S$)": st.column_config.NumberColumn(
                    format="S$%,.0f"
                ),
                "Mean Predicted Employment Rate (%)": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            },
        )

    elif page == "Programme Tables":
        st.header("Schools and Degrees within Each Cluster")
        profiles = [
            profile for profile in PROFILE_ORDER
            if profile in set(predictions["outcome_profile_2026"])
        ]
        selected_profiles = st.multiselect(
            "Filter by predicted outcome profile",
            options=profiles,
            default=profiles,
        )

        filtered = predictions[
            predictions["outcome_profile_2026"].isin(selected_profiles)
        ].copy()
        display_table = programme_table(filtered).sort_values(
            ["Cluster", "University", "School", "Degree"]
        )

        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Predicted 2026 Median Salary (S$)": st.column_config.NumberColumn(
                    format="S$%,.0f"
                ),
                "Predicted 2026 Employment Rate (%)": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            },
        )
        st.download_button(
            "Download filtered programme table as CSV",
            data=display_table.to_csv(index=False).encode("utf-8"),
            file_name="predicted_2026_programmes_by_profile.csv",
            mime="text/csv",
        )

        for profile in profiles:
            profile_data = filtered[
                filtered["outcome_profile_2026"] == profile
            ].copy()
            if profile_data.empty:
                continue
            with st.expander(f"{profile} — {len(profile_data)} programmes"):
                st.dataframe(
                    programme_table(profile_data).sort_values(
                        "Predicted 2026 Median Salary (S$)",
                        ascending=False,
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Predicted 2026 Median Salary (S$)": st.column_config.NumberColumn(
                            format="S$%,.0f"
                        ),
                        "Predicted 2026 Employment Rate (%)": st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),
                    },
                )

    else:
        st.header("Model Evaluation and Limitations")
        st.write(
            "The model was evaluated using a time-based design: input years 2013–2023 "
            "were used for training, and 2024 programme records were used to predict "
            "observed 2025 outcomes."
        )
        st.dataframe(
            evaluation_metrics,
            width="stretch",
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.2f"),
                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.markdown(
            """
            **Interpretation**

            - Salary MAE and RMSE are measured in **Singapore dollars**.
            - Employment-rate MAE and RMSE are measured in **percentage points**.
            - R² is unitless and describes how much test-set variation is explained by the model.

            **Limitations**

            - Results describe **programme-level historical patterns**, not individual outcomes.
            - The models do not observe personal skills, work experience, internships, job-search strategy, or economic shocks.
            - The 2026 cluster profiles group model predictions; they are not confirmed outcomes or programme rankings.
            """
        )


if __name__ == "__main__":
    app()
