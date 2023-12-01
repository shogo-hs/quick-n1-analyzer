import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def plot_payment_history(
    user_data: pd.DataFrame,
    datetime_col: str,
    price_col: str,
    label_col: str,
) -> None:
    """
    対象のユーザーの決済履歴をグラフに描写する。（x軸:決済日時、y軸:決済金額、凡例:正常or不正）
    """

    # 日付データをmatplotlibが解釈できる形式に変換。
    dates = mdates.date2num(user_data[datetime_col])

    # 折れ線グラフのセグメントを作成し、ラベルによって色を決定。
    points = np.array([dates, user_data[price_col]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 通常の取引と不正取引を区別するための色付け。
    normalsegments = segments[user_data[label_col].to_numpy()[:-1] == 0]
    fraudsegments = segments[user_data[label_col].to_numpy()[:-1] == 1]

    normal_lc = LineCollection(
        normalsegments, colors="blue", linewidth=2, label="Normal"
    )
    fraud_lc = LineCollection(fraudsegments, colors="red", linewidth=2, label="Fraud")

    plt.figure(figsize=(10, 4))
    plt.gca().add_collection(normal_lc)
    plt.gca().add_collection(fraud_lc)
    plt.xlim(dates.min(), dates.max())
    plt.ylim(user_data[price_col].min(), user_data[price_col].max())

    # X軸に日付のフォーマットを設定
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # X軸の日付が重ならないように調整
    plt.gcf().autofmt_xdate()

    # タイトルと軸ラベルの設定
    plt.title(f"Payment History")
    plt.xlabel("Date")
    plt.ylabel("Payment Amount")

    # 凡例を追加
    plt.legend(handles=[normal_lc, fraud_lc])

    # グリッドとレイアウトの調整
    plt.grid(True)
    plt.tight_layout()
    plt.show()
