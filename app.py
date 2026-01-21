import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "tlip_like_consignments_5000.jsonl"
MODEL_PATH = BASE_DIR / "models" / "log_reg_pipeline.pkl"
IMAGE_PATH = BASE_DIR / "assets" / "horticulture-main-1200x900.jpg"


@st.cache_resource
def load_model():
    # Cache the pickled pipeline so it is loaded only once per session.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    # Cache the dataset for dropdown population; not used for predictions.
    if not DATA_PATH.exists():
        return None
    return pd.read_json(DATA_PATH, lines=True)


def options_for(df: pd.DataFrame, column: str):
    # Provide sorted unique values for select boxes.
    if df is None or column not in df.columns:
        return []
    values = df[column].dropna().unique().tolist()
    return sorted(values)


@st.cache_data
def get_global_feature_importance(_model) -> pd.DataFrame:
    # Summarize global logistic regression coefficients for interpretability.
    feature_names = _model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = _model.named_steps["classifier"].coef_[0]

    importance_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    return importance_df


def get_prediction_contributions(model, input_df: pd.DataFrame) -> pd.DataFrame:
    # Compute per-feature contributions for the single prediction.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    transformed = model.named_steps["preprocessor"].transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = transformed.flatten() * model.named_steps["classifier"].coef_[0]

    return pd.DataFrame(
        {"feature": feature_names, "contribution": contributions}
    ).sort_values("contribution", ascending=False)


def assign_risk_tiers(values: pd.Series) -> pd.Series:
    # Bucket values into Low/Medium/High impact tiers using tertiles.
    tiers = pd.qcut(values.rank(method="first"), 3, labels=["Low", "Medium", "High"])
    return tiers.astype(str)


def main() -> None:
    # Build the Streamlit UI and run predictions on demand.
    st.set_page_config(page_title="Clearance Delay Predictor", layout="wide")
    st.title("Clearance Delay Risk Prediction")

    description_col, image_col = st.columns([2, 1])
    with description_col:
        st.write(
            "Overview\n\nDelays in horticultural supply chains can quickly lead to financial losses, product spoilage, and missed market opportunities. This project addresses that challenge by providing a simple, data-driven way to assess whether an export consignment is likely to be delayed during clearance. By analyzing key shipment details, documentation readiness, and operational conditions, the system helps identify delay risks early—before issues escalate.The tool is designed to be easy to use. Users enter basic consignment information through a short form and click Predict. The system then evaluates the inputs using a trained machine-learning model and returns a clear indication of whether the shipment is likely to be delayed, along with a confidence score. This turns complex analytics into an accessible decision-support tool, enabling exporters and logistics teams to take proactive action and improve supply-chain reliability."
        )
        st.info(
            "Fill in the shipment details, then click **Predict delay risk** to "
            "see the probability and top contributing factors."
        )
    with image_col:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), use_column_width=True)

    model_payload = load_model()
    model = model_payload["model"]
    features = model_payload["features"]

    df = load_data()

    st.subheader("Shipment Inputs")
    input_col, factor_col = st.columns([2, 1])

    with input_col:
        st.markdown("**Shipment Details**")
        shipment_mode = st.selectbox(
            "Shipment mode",
            options=options_for(df, "shipment_mode") or ["Unknown"],
        )
        commodity = st.selectbox(
            "Commodity",
            options=options_for(df, "commodity") or ["Unknown"],
        )
        hs_code = st.selectbox(
            "HS code",
            options=options_for(df, "hs_code") or ["Unknown"],
        )
        origin_country = st.selectbox(
            "Origin country",
            options=options_for(df, "origin_country") or ["Unknown"],
        )
        destination_country = st.selectbox(
            "Destination country",
            options=options_for(df, "destination_country") or ["Unknown"],
        )
        exporter_profile = st.selectbox(
            "Exporter profile",
            options=options_for(df, "exporter_profile") or ["Unknown"],
        )

        st.markdown("**Documentation and Operations**")
        doc_completeness_score = st.number_input(
            "Document completeness score",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )
        missing_docs_proxy = st.number_input(
            "Missing docs proxy", min_value=0.0, value=0.0, step=1.0
        )
        doc_amendments = st.number_input(
            "Document amendments", min_value=0.0, value=0.0, step=1.0
        )
        congestion_index = st.number_input(
            "Congestion index", min_value=0.0, value=0.5, step=0.05
        )
        gross_weight_kg = st.number_input(
            "Gross weight (kg)", min_value=0.0, value=1000.0, step=10.0
        )
        declared_value_usd = st.number_input(
            "Declared value (USD)", min_value=0.0, value=10000.0, step=100.0
        )
        is_weekend_created = st.selectbox(
            "Created on weekend?",
            options=[0, 1],
            format_func=lambda v: "Yes" if v == 1 else "No",
        )

    with factor_col:
        st.markdown("**Key Delay Drivers (Global)**")
        importance_df = get_global_feature_importance(model)
        positive = importance_df.head(8).reset_index(drop=True)
        negative = importance_df.tail(8).sort_values("coefficient").reset_index(drop=True)
        positive["impact_tier"] = assign_risk_tiers(positive["abs_coefficient"])
        negative["impact_tier"] = assign_risk_tiers(negative["abs_coefficient"])

        st.write("Higher risk factors")
        st.dataframe(
            positive[["feature", "impact_tier"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            positive.set_index("feature")["abs_coefficient"],
            use_container_width=True,
        )
        st.write("Lower risk factors")
        st.dataframe(
            negative[["feature", "impact_tier"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            negative.set_index("feature")["abs_coefficient"],
            use_container_width=True,
        )

    input_df = pd.DataFrame(
        [
            {
                "shipment_mode": shipment_mode,
                "commodity": commodity,
                "hs_code": hs_code,
                "origin_country": origin_country,
                "destination_country": destination_country,
                "exporter_profile": exporter_profile,
                "doc_completeness_score": doc_completeness_score,
                "missing_docs_proxy": missing_docs_proxy,
                "doc_amendments": doc_amendments,
                "congestion_index": congestion_index,
                "is_weekend_created": is_weekend_created,
                "gross_weight_kg": gross_weight_kg,
                "declared_value_usd": declared_value_usd,
            }
        ]
    )

    input_df = input_df[features]

    if st.button("Predict delay risk"):
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])

        label = "Delayed" if prediction == 1 else "On time"
        st.metric("Prediction", label)
        st.metric("Delay probability", f"{probability:.2%}")

        st.subheader("Top contributors for this prediction")
        contributions_df = get_prediction_contributions(model, input_df)
        contributions_df["abs_contribution"] = contributions_df["contribution"].abs()
        top_positive = contributions_df.head(6).reset_index(drop=True)
        top_negative = contributions_df.tail(6).sort_values("contribution").reset_index(
            drop=True
        )
        top_positive["impact_tier"] = assign_risk_tiers(
            top_positive["abs_contribution"]
        )
        top_negative["impact_tier"] = assign_risk_tiers(
            top_negative["abs_contribution"]
        )

        pos_col, neg_col = st.columns(2)
        with pos_col:
            st.write("Increase delay risk")
            st.dataframe(
                top_positive[["feature", "impact_tier"]],
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                top_positive.set_index("feature")["abs_contribution"],
                use_container_width=True,
            )
        with neg_col:
            st.write("Reduce delay risk")
            st.dataframe(
                top_negative[["feature", "impact_tier"]],
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                top_negative.set_index("feature")["abs_contribution"],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
