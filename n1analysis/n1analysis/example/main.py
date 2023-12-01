from datetime import datetime, timedelta

import pandas as pd

from n1analysis.binary_classification.misclassification_analysis import (
    MisClassifiedInspector,
)
from n1analysis.datasets.payment_transaction import generate_transaction_data

if __name__ == "__main__":
    df_normal = generate_transaction_data(
        num_entries=1000,
        user_id_range=(1000, 1011),
        merchant_id_range=(1, 10),
        date_range=(
            datetime.now() + timedelta(days=-365),
            datetime.now() + timedelta(days=-180),
        ),
        payment_methods=["credit_cart", "cash"],
        fraud_percentage=0.0,
    )

    df_fraud = generate_transaction_data(
        num_entries=100,
        user_id_range=(1000, 1011),
        merchant_id_range=(1, 10),
        date_range=(
            datetime.now() + timedelta(days=-179),
            datetime.now() + timedelta(days=-175),
        ),
        payment_methods=["credit_cart", "cash"],
        fraud_percentage=1.0,
    )

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df["use_dt"] = pd.to_datetime(df["use_dt"])

    inspector = MisClassifiedInspector(
        dataset=df,
        user_id_col="user_id",
        price_col="price",
        datetime_col="use_dt",
        prob_col="probability",
        label_col="label",
    )

    inspector.run()
