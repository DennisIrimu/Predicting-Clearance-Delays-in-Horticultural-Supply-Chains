import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_FEATURES = [
    "shipment_mode",
    "commodity",
    "hs_code",
    "origin_country",
    "destination_country",
    "exporter_profile",
    "doc_completeness_score",
    "missing_docs_proxy",
    "doc_amendments",
    "congestion_index",
    "is_weekend_created",
    "gross_weight_kg",
    "declared_value_usd",
]

CATEGORICAL_FEATURES = [
    "shipment_mode",
    "commodity",
    "hs_code",
    "origin_country",
    "destination_country",
    "exporter_profile",
]

NUMERIC_FEATURES = [
    "doc_completeness_score",
    "missing_docs_proxy",
    "doc_amendments",
    "congestion_index",
    "gross_weight_kg",
    "declared_value_usd",
]

BINARY_FEATURES = ["is_weekend_created"]


def build_pipeline() -> Pipeline:
    # Preprocess categorical/numeric/binary features consistently for training and inference.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )

    classifier = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Bundle preprocessing + classifier so downstream code only needs raw inputs.
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train(data_path: Path) -> dict:
    # Load training data and fit the pipeline on the baseline feature set.
    df = pd.read_json(data_path, lines=True)
    X = df[BASELINE_FEATURES]
    y = df["delayed_flag"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # Return both the fitted pipeline and the ordered feature list for inference.
    return {"model": pipeline, "features": BASELINE_FEATURES}


def main() -> None:
    # Parse CLI args, train the model, and persist it to disk.
    parser = argparse.ArgumentParser(
        description="Train and pickle the elastic-net logistic regression pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="tlip_like_consignments_5000.jsonl",
        help="Path to the JSONL dataset.",
    )
    parser.add_argument(
        "--model-path",
        default="models/log_reg_pipeline.pkl",
        help="Where to save the pickled model.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = train(data_path)

    with model_path.open("wb") as file:
        pickle.dump(payload, file)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
