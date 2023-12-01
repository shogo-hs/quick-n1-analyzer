import random
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def generate_transaction_data(
    num_entries: int,
    user_id_range: Tuple[int, int],
    merchant_id_range: Tuple[int, int],
    date_range: Tuple[datetime, datetime],
    payment_methods: List[str],
    fraud_percentage: float = 0.01,
) -> pd.DataFrame:
    """
    不正利用を含む決済トランザクションの疑似データを生成する関数。
    各取引には 'label' と 'probability' の2つのカラムが含まれ、
    'label' は不正利用を示す（0または1）、'probability' はその取引が不正である確率を示します。

    Args:
    num_entries (int): 生成するデータエントリの数。
    user_id_range (Tuple[int, int]): ユーザIDの最小値と最大値の範囲。
    merchant_id_range (Tuple[int, int]): 加盟店IDの最小値と最大値の範囲。
    date_range (Tuple[datetime, datetime]): 決済日時の開始日と終了日。
    payment_methods (List[str]): 利用可能な支払い方法のリスト。
    fraud_percentage (float): 全取引における不正利用の割合（デフォルトは1%）。

    Returns:
    pd.DataFrame: 生成された決済トランザクション、不正利用ラベル、確率を含むデータセット。
    """
    data = []
    num_frauds = int(num_entries * fraud_percentage)

    for i in range(num_entries):
        user_id = str(random.randint(*user_id_range))
        merchant_id = str(random.randint(*merchant_id_range))
        date_time = date_range[0] + (date_range[1] - date_range[0]) * random.random()
        amount = round(random.uniform(10, 500), 0)
        discount = round(amount * random.uniform(0, 0.3), 0)
        payment_method = random.choice(payment_methods)

        is_fraud = i < num_frauds
        if is_fraud:
            amount = round(random.uniform(300, 500), 0)
            probability = round(random.uniform(0.4, 1.0), 3)
        else:
            probability = round(random.uniform(0.0, 0.6), 3)

        label = int(is_fraud)
        data.append(
            [
                user_id,
                merchant_id,
                date_time,
                amount,
                discount,
                payment_method,
                label,
                probability,
            ]
        )

    random.shuffle(data)

    return pd.DataFrame(
        data,
        columns=[
            "user_id",
            "shop_id",
            "use_dt",
            "price",
            "discount",
            "pay_method",
            "label",
            "probability",
        ],
    )


# この関数を使用してデータを生成する
# df = generate_transaction_data(...)
