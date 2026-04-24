from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("eda_outputs")
REDUCED_HEATMAP_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (evaluations)",
    "Unemployment rate",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def print_basic_overview(df: pd.DataFrame) -> None:
    print("Dataset overview")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    missing = df.isna().sum()
    if (missing > 0).any():
        print(missing[missing > 0].to_string())
    else:
        print("No missing values")

    print(f"\nDuplicated rows: {df.duplicated().sum()}")


def print_target_summary(df: pd.DataFrame) -> None:
    print("\nTarget distribution")
    counts = df["Target"].value_counts()
    percentages = df["Target"].value_counts(normalize=True).mul(100).round(2)
    summary = pd.DataFrame({"count": counts, "percent": percentages})
    print(summary.to_string())


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    low_cardinality_cols = [
        col for col in numeric_cols if 2 < df[col].nunique() <= 20 and col != "Target"
    ]
    continuous_cols = [
        col for col in numeric_cols if df[col].nunique() > 20 and col not in binary_cols
    ]

    key_numeric = [
        "Age at enrollment",
        "Previous qualification (grade)",
        "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    key_binary = [
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
        "Displaced",
        "Educational special needs",
        "Daytime/evening attendance",
    ]
    key_categorical = [
        "Course",
        "Application mode",
        "Application order",
        "Previous qualification",
        "Marital status",
        "Nacionality",
    ]

    groups = {
        "numeric": numeric_cols,
        "binary": [col for col in key_binary if col in df.columns]
        + [col for col in binary_cols if col not in key_binary],
        "low_cardinality": low_cardinality_cols,
        "continuous": continuous_cols,
        "key_numeric": [col for col in key_numeric if col in df.columns],
        "key_categorical": [col for col in key_categorical if col in df.columns],
    }
    return groups


def save_target_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    order = df["Target"].value_counts().index.tolist()
    if sns is not None:
        ax = sns.countplot(data=df, x="Target", order=order)
    else:
        counts = df["Target"].value_counts().reindex(order)
        ax = counts.plot(kind="bar", color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Target Distribution")
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("Count")
    if hasattr(ax, "containers"):
        for container in ax.containers:
            ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200)
    plt.close()


def save_numeric_distributions(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        plt.figure(figsize=(8, 5))
        if sns is not None:
            sns.histplot(df[col], kde=True, bins=30)
        else:
            plt.hist(df[col], bins=30, edgecolor="black", alpha=0.8)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        safe_name = (
            col.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(OUTPUT_DIR / f"dist_{safe_name}.png", dpi=200)
        plt.close()


def save_numeric_vs_target(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        plt.figure(figsize=(8, 5))
        if sns is not None:
            sns.boxplot(
                data=df, x="Target", y=col, order=["Dropout", "Enrolled", "Graduate"]
            )
        else:
            order = ["Dropout", "Enrolled", "Graduate"]
            data = [df.loc[df["Target"] == target, col] for target in order]
            plt.boxplot(data, tick_labels=order)
        plt.title(f"{col} by Target")
        plt.xlabel("Target")
        plt.ylabel(col)
        plt.tight_layout()
        safe_name = (
            col.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(OUTPUT_DIR / f"box_{safe_name}_by_target.png", dpi=200)
        plt.close()


def save_binary_vs_target(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        proportions = pd.crosstab(df[col], df["Target"], normalize="index")
        proportions = proportions.reindex(sorted(proportions.index), axis=0)
        proportions.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set2")
        plt.title(f"Target Proportions by {col}")
        plt.xlabel(col)
        plt.ylabel("Proportion")
        plt.legend(title="Target")
        plt.tight_layout()
        safe_name = (
            col.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(OUTPUT_DIR / f"stacked_{safe_name}_by_target.png", dpi=200)
        plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    plt.figure(figsize=(16, 12))
    if sns is not None:
        sns.heatmap(corr, cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap for Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_reduced_correlation_heatmap(df: pd.DataFrame) -> None:
    selected = [col for col in REDUCED_HEATMAP_FEATURES if col in df.columns]
    corr = df[selected].corr()
    plt.figure(figsize=(11, 8))
    if sns is not None:
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    else:
        plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Reduced Correlation Heatmap of Selected Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap_reduced.png", dpi=220)
    plt.close()


def save_pairplot(df: pd.DataFrame) -> None:
    pairplot_cols = [
        "Target",
        "Admission grade",
        "Previous qualification (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]
    available_cols = [col for col in pairplot_cols if col in df.columns]
    pair_df = df[available_cols].copy()
    sample_size = min(len(pair_df), 700)
    pair_df = pair_df.sample(sample_size, random_state=42)
    grid = sns.pairplot(
        pair_df,
        hue="Target",
        corner=True,
        diag_kind="hist",
        height=2.4,
        plot_kws={"alpha": 0.6, "s": 28},
        diag_kws={"bins": 20},
    )
    grid.fig.set_size_inches(16, 14)
    for ax_row in grid.axes:
        for ax in ax_row:
            if ax is None:
                continue
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
    grid.fig.subplots_adjust(
        top=0.95, bottom=0.08, left=0.08, right=0.98, hspace=0.12, wspace=0.12
    )
    grid.fig.suptitle("Pairplot of Key Predictive Features", y=0.995, fontsize=14)
    grid.savefig(OUTPUT_DIR / "key_feature_pairplot.png", dpi=180)
    plt.close("all")


def main() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("ggplot")
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    df = clean_columns(df)

    groups = get_column_groups(df)

    print_basic_overview(df)
    print_target_summary(df)

    save_target_plot(df)
    save_numeric_distributions(df, groups["key_numeric"])
    save_numeric_vs_target(df, groups["key_numeric"])
    save_binary_vs_target(df, groups["binary"][:8])
    save_correlation_heatmap(df)
    save_reduced_correlation_heatmap(df)
    save_pairplot(df)


if __name__ == "__main__":
    main()
