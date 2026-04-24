from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATA_PATH = Path("data.csv")
TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
RESULTS_PATH = Path("model_comparison_results.csv")
BEST_CONFUSION_MATRIX_PATH = Path("best_model_confusion_matrix.csv")
TUNING_RESULTS_PATH = Path("random_forest_tuning_comparison.csv")
FEATURE_IMPORTANCE_PATH = Path("random_forest_feature_importance.csv")
FEATURE_IMPORTANCE_PLOT_PATH = Path("random_forest_feature_importance_top10.png")
DROP_COLUMNS = [
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
]

CONTINUOUS_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = clean_columns(df)
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def infer_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    feature_df = df.drop(columns=[TARGET_COLUMN])
    numeric_columns = feature_df.select_dtypes(include="number").columns.tolist()

    binary_features = [col for col in numeric_columns if feature_df[col].nunique() == 2]
    continuous_features = [
        col for col in CONTINUOUS_FEATURES if col in feature_df.columns
    ]
    categorical_features = [
        col
        for col in numeric_columns
        if col not in set(binary_features) and col not in set(continuous_features)
    ]

    return {
        "all_features": feature_df.columns.tolist(),
        "binary": binary_features,
        "continuous": continuous_features,
        "categorical": categorical_features,
    }


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index, name=TARGET_COLUMN)
    return y_encoded, encoder


def build_linear_preprocessor(groups: dict[str, list[str]]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), groups["continuous"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                groups["categorical"] + groups["binary"],
            ),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(groups: dict[str, list[str]]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("passthrough_all", "passthrough", groups["all_features"]),
        ],
        remainder="drop",
    )


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_model_ready_pipelines(df: pd.DataFrame) -> dict[str, object]:
    groups = infer_feature_groups(df)
    X = df.drop(columns=[TARGET_COLUMN])
    y, y_encoder = encode_target(df[TARGET_COLUMN])
    X_train, X_test, y_train, y_test = split_data(X, y)

    linear_preprocessor = build_linear_preprocessor(groups)
    tree_preprocessor = build_tree_preprocessor(groups)

    return {
        "feature_groups": groups,
        "label_encoder": y_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "linear_preprocessor": linear_preprocessor,
        "tree_preprocessor": tree_preprocessor,
    }


def build_model_pipelines(
    linear_preprocessor: ColumnTransformer, tree_preprocessor: ColumnTransformer
) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    results = []
    best_model_name = None
    best_predictions = None
    best_macro_f1 = -1.0

    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics_row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "weighted_precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "weighted_recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }
        results.append(metrics_row)

        if metrics_row["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics_row["macro_f1"]
            best_model_name = model_name
            best_predictions = y_pred

    results_df = pd.DataFrame(results).sort_values(
        by=["macro_f1", "weighted_f1", "accuracy"], ascending=False
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    if best_model_name is not None and best_predictions is not None:
        cm = confusion_matrix(y_test, best_predictions)
        cm_df = pd.DataFrame(
            cm,
            index=[f"actual_{label}" for label in label_encoder.classes_],
            columns=[f"predicted_{label}" for label in label_encoder.classes_],
        )
        cm_df.to_csv(BEST_CONFUSION_MATRIX_PATH)

    return results_df


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def tune_random_forest(
    tree_preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    baseline_pipeline = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced"],
    }

    tuned_search = RandomizedSearchCV(
        estimator=Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )

    baseline_pipeline.fit(X_train, y_train)
    baseline_predictions = baseline_pipeline.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_predictions)

    tuned_search.fit(X_train, y_train)
    tuned_predictions = tuned_search.best_estimator_.predict(X_test)
    tuned_metrics = compute_metrics(y_test, tuned_predictions)

    comparison_df = pd.DataFrame(
        [
            {
                "model_version": "untuned_random_forest",
                "cv_best_macro_f1": None,
                **baseline_metrics,
            },
            {
                "model_version": "tuned_random_forest",
                "cv_best_macro_f1": tuned_search.best_score_,
                **tuned_metrics,
            },
        ]
    )
    comparison_df.to_csv(TUNING_RESULTS_PATH, index=False)

    return comparison_df


def export_random_forest_feature_importance(
    tree_preprocessor: ColumnTransformer,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    feature_pipeline = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    feature_pipeline.fit(X_train, y_train)
    importances = feature_pipeline.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    top_10 = importance_df.head(10).sort_values(by="importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_10["feature"], top_10["importance"], color="#4c72b0")
    plt.title("Top 10 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=220)
    plt.close()

    return importance_df


def main() -> None:
    df = load_dataset()
    assets = build_model_ready_pipelines(df)
    models = build_model_pipelines(
        assets["linear_preprocessor"], assets["tree_preprocessor"]
    )
    evaluate_models(
        models,
        assets["X_train"],
        assets["X_test"],
        assets["y_train"],
        assets["y_test"],
        assets["label_encoder"],
    )

    tune_random_forest(
        assets["tree_preprocessor"],
        assets["X_train"],
        assets["X_test"],
        assets["y_train"],
        assets["y_test"],
        assets["label_encoder"],
    )

    export_random_forest_feature_importance(
        assets["tree_preprocessor"],
        assets["feature_groups"]["all_features"],
        assets["X_train"],
        assets["y_train"],
    )


if __name__ == "__main__":
    main()
