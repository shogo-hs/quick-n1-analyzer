import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def calculate_classification_types(
    dataset: pd.DataFrame, prob_col: str, label_col: str, threshold: float
) -> pd.DataFrame:
    dataset["predicted_label"] = (dataset[prob_col] >= threshold).astype(int)
    dataset.loc[
        (dataset[label_col] == 1) & (dataset["predicted_label"] == 1),
        "classification_type",
    ] = "TP"
    dataset.loc[
        (dataset[label_col] == 0) & (dataset["predicted_label"] == 0),
        "classification_type",
    ] = "TN"
    dataset.loc[
        (dataset[label_col] == 0) & (dataset["predicted_label"] == 1),
        "classification_type",
    ] = "FP"
    dataset.loc[
        (dataset[label_col] == 1) & (dataset["predicted_label"] == 0),
        "classification_type",
    ] = "FN"

    return dataset


def calculate_classification_types_spark(
    dataset: SparkDataFrame, prob_col: str, label_col: str, threshold: float
) -> SparkDataFrame:
    # モデルの確率閾値に基づいて予測ラベルを計算
    dataset = dataset.withColumn(
        "predicted_label", (F.col(prob_col) >= threshold).cast("int")
    )

    # 分類のタイプを計算
    dataset = dataset.withColumn(
        "classification_type",
        F.when((F.col(label_col) == 1) & (F.col("predicted_label") == 1), "TP")
        .when((F.col(label_col) == 0) & (F.col("predicted_label") == 0), "TN")
        .when((F.col(label_col) == 0) & (F.col("predicted_label") == 1), "FP")
        .when((F.col(label_col) == 1) & (F.col("predicted_label") == 0), "FN"),
    )
    return dataset


def get_classification_data_by_type(
    df: pd.DataFrame, type_col: str, extract_type: str
) -> pd.DataFrame:
    return df[df[type_col] == extract_type].reset_index(drop=True)


def get_classification_data_by_type_spark(
    df: SparkDataFrame, type_col: str, extract_type: str
) -> SparkDataFrame:
    return df.filter(F.col(type_col) == extract_type)
