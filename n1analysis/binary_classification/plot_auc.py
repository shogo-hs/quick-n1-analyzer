import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Pandas DataFrame用の精度と再現率の計算関数
def calculate_precision_recall(df, label_col, probability_col, decimal_places=3):
    """
    Calculates precision and recall for a Pandas DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the true labels and predicted probabilities
    label_col (str): Column name for true labels
    probability_col (str): Column name for predicted probabilities
    decimal_places (int): Number of decimal places to round the probabilities to

    Returns:
    pd.DataFrame: DataFrame containing the recall and precision values.
    """
    # Round the probabilities
    df["probability_round"] = df[probability_col].round(decimal_places)

    # Total number of positives (P)
    P = df[label_col].sum()

    # Sort by predicted probabilities in descending order
    df_sorted = df.sort_values(by="probability_round", ascending=False).reset_index(
        drop=True
    )

    # Calculate cumulative true positives and false positives
    df_sorted["TP_cumulative"] = df_sorted[label_col].cumsum()
    df_sorted["FP_cumulative"] = df_sorted["TP_cumulative"] - df_sorted[label_col]

    # Calculate precision and recall values
    df_sorted["precision"] = df_sorted["TP_cumulative"] / (df_sorted.index + 1)
    df_sorted["recall"] = df_sorted["TP_cumulative"] / P

    return df_sorted[["recall", "precision"]]


# PySpark DataFrame用の精度と再現率の計算関数
def calculate_precision_recall_spark(df, label_col, probability_col, decimal_places=3):
    """
    Calculates precision and recall for a PySpark DataFrame.

    Parameters:
    df (pyspark.sql.DataFrame): DataFrame containing the true labels and predicted probabilities
    label_col (str): Column name for true labels
    probability_col (str): Column name for predicted probabilities
    decimal_places (int): Number of decimal places to round the probabilities to

    Returns:
    pyspark.sql.DataFrame: DataFrame containing the recall and precision values.
    """
    # Round the probabilities to the specified number of decimal places
    df = df.withColumn("probability_round", F.bround(probability_col, decimal_places))

    # Total number of positives (P)
    P = df.select(F.sum(label_col)).collect()[0][0]

    # Define the window specification for cumulative sum ordered by rounded probability
    windowSpec = Window.orderBy(F.col("probability_round").desc())

    # Calculate cumulative true positives and false positives
    df = df.withColumn("TP_cumulative", F.sum(label_col).over(windowSpec))
    df = df.withColumn("FP", (F.lit(1) - F.col(label_col)))
    df = df.withColumn("FP_cumulative", F.sum("FP").over(windowSpec))

    # Calculate precision and recall
    df = df.withColumn(
        "precision",
        F.col("TP_cumulative") / (F.col("TP_cumulative") + F.col("FP_cumulative")),
    )
    df = df.withColumn("recall", F.col("TP_cumulative") / F.lit(P))

    return df.select("recall", "precision")
