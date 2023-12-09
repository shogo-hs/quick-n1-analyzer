from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from binary_classification.classification_utils import (
    calculate_classification_types,
    calculate_classification_types_spark,
    get_classification_data_by_type,
    get_classification_data_by_type_spark,
)
from binary_classification.plot_auc import (
    calculate_precision_recall,
    calculate_precision_recall_spark,
)
from binary_classification.plot_transaction import plot_payment_history
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.plotting import register_matplotlib_converters
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

register_matplotlib_converters()


class MisClassifiedInspector:
    """
    誤分類されたトランザクションに対してユーザー単位でN1分析を実行するためのクラス。
    ユーザーが指定した閾値を基に予測ラベルを算出し、その予測ラベルと実際のラベルを比較して混同行列を生成。
    ユーザーの決済履歴をデータフレームとして可視化。
    ユーザーの決済日時と金額をグラフ化。
    """

    def __init__(
        self,
        dataset: Union[pd.DataFrame, SparkDataFrame],
        user_id_col: str,
        price_col: str = "price",
        datetime_col: str = "use_dt",
        prob_col: str = "probability",
        label_col: str = "label",
        threshold: float = 0.5,
        spark: SparkSession = None,
    ) -> None:
        """
        与えられたデータセットとカラム名でMisClassifiedInspectorを初期化します。

        Args:
            dataset (Union[pd.DataFrame, SparkDataFrame]): 分析対象のデータセット。
                Pandas DataFrameまたはSpark DataFrameのいずれかを指定します。
            user_id_col (str): ユーザーIDを表すカラム名。
            price_col (str): 価格を表すカラム名。デフォルトは "price"。
            datetime_col (str): 日時を表すカラム名。デフォルトは "use_dt"。
            prob_col (str): 確率を表すカラム名。デフォルトは "probability"。
            label_col (str): ラベルを表すカラム名。デフォルトは "label"。
            threshold (float): 分類のための確率の閾値。デフォルトは 0.5。
            spark(SparkSession): Spark DataFrameを扱う場合はSpark Sessionを指定します。
                datasetがPandas DataFrameの場合、この引数は無視されます。デフォルトはNone。
        """
        self.dataset = dataset
        self.user_id_col = user_id_col
        self.price_col = price_col
        self.datetime_col = datetime_col
        self.prob_col = prob_col
        self.label_col = label_col
        self.threshold = threshold
        self.misclassified_data = None
        self.spark = spark
        self.columns = None
        self.user_data = None

    def get_misclassified_data(self) -> None:
        """
        誤分類されたデータを識別し、データセットに列を追加する。
        """
        if self.spark is None:
            self.dataset = calculate_classification_types(
                self.dataset, self.prob_col, self.label_col, self.threshold
            )
            self.misclassified_data = self.dataset[
                self.dataset["classification_type"].isin(["FP", "FN"])
            ]
        else:
            self.dataset = calculate_classification_types_spark(
                self.dataset, self.prob_col, self.label_col, self.threshold
            )

            self.misclassified_data = self.dataset.filter(
                F.col("classification_type").isin(["FP", "FN"])
            )
        self.columns = list(self.dataset.columns)  # 更新

    def get_misclassified_data_by_type(
        self, type: str
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        if self.spark is None:
            return get_classification_data_by_type(
                self.misclassified_data, "classification_type", type
            )

        else:
            return get_classification_data_by_type_spark(
                self.misclassified_data, "classification_type", "FP"
            )

    def get_selected_user_data(self, user_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        特定のユーザーIDに該当するデータを返す。

        Args:
            user_id (str): ユーザーID
            df (pd.DataFrame): 検索対象のデータフレーム

        Returns:
            pd.DataFrame: 指定されたユーザーIDに該当するデータ
        """
        if self.spark is None:
            return df[df[self.user_id_col] == user_id].sort_values(self.datetime_col)
        else:
            return (
                df.filter(F.col(self.user_id_col) == user_id)
                .orderBy(self.datetime_col)
                .toPandas()
            )

    def create_confusion_matrix(self) -> pd.DataFrame:
        """
        データセットから混同行列を作成し、真陽性（TP）、偽陽性（FP）、真陰性（TN）、偽陰性（FN）の各数値を含むデータフレームを返す。

        Returns:
            pd.DataFrame: TP、FP、TN、FNのカウントを含む混同行列のデータフレーム。
        """

        # 各値のカウント
        if self.spark is None:
            value_counts = self.dataset["classification_type"].value_counts()
        else:
            value_counts_df = self.dataset.groupBy("classification_type").count()

            # PySpark DataFrameをPandas DataFrameに変換
            value_counts_pandas = value_counts_df.toPandas()

            # Pandas DataFrameをシリーズに変換し、'classification_type' をインデックスに設定
            value_counts = value_counts_pandas.set_index("classification_type")["count"]

        # 新しいデータフレームの作成
        count_df = pd.DataFrame(value_counts)
        count_df.reset_index(inplace=True)
        count_df.columns = ["Item", "Value"]

        # カウント結果のデータフレームを指定の順序で並べ替え
        ordered_values = ["TP", "FP", "TN", "FN"]
        count_df_ordered = (
            count_df.set_index("Item").reindex(ordered_values).reset_index()
        )
        count_df_ordered.fillna(0, inplace=True)  # 0の値を持つ行を0で埋める

        # Recall と Precision の計算
        TP = count_df_ordered[count_df_ordered["Item"] == "TP"]["Value"].values[0]
        FP = count_df_ordered[count_df_ordered["Item"] == "FP"]["Value"].values[0]
        FN = count_df_ordered[count_df_ordered["Item"] == "FN"]["Value"].values[0]

        # 0で割ることを避けるためのチェック
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0

        # Recall と Precision をデータフレームに追加
        additional_rows = pd.DataFrame(
            {
                "Item": ["Recall", "Precision"],
                "Value": ["{:.4f}".format(recall), "{:.4f}".format(precision)],
            }
        )

        return pd.concat([count_df_ordered, additional_rows], ignore_index=True)

    def get_confusion_matrix_lists(self) -> tuple:
        """
        混同行列のデータとヘッダーをリスト形式で生成する。

        Returns:
            tuple: データのリストとヘッダーのリストを含むタプル。
        """
        confusion_matrix_df = self.create_confusion_matrix()
        confusion_matrix_data = confusion_matrix_df.values.tolist()
        confusion_matrix_headers = confusion_matrix_df.columns.tolist()
        return confusion_matrix_data, confusion_matrix_headers

    def plot_pr_auc(self):
        if self.spark is None:
            pr_df = calculate_precision_recall(
                self.dataset, self.label_col, self.prob_col
            )
        else:
            pr_df = calculate_precision_recall_spark(
                self.dataset, self.label_col, self.prob_col
            )

        # グラフサイズは変更せずに、ここでfigsizeをそのままにします。
        fig, ax = plt.subplots(figsize=(2, 2))  # このサイズは小さすぎる可能性がありますが、要求に応じてそのままにします。

        # グラフのデータをプロット
        ax.plot(pr_df["recall"], pr_df["precision"])

        # 軸ラベルのフォントサイズをさらに小さくします。
        ax.set_xlabel("Recall", fontsize=7)
        ax.set_ylabel("Precision", fontsize=7)

        # タイトルのフォントサイズを小さくし、余白を調整します。
        ax.set_title("Precision-Recall Curve", fontsize=8)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

        # ティックのフォントサイズを調整
        ax.tick_params(axis="both", which="major", labelsize=6)

        return fig

    def export_to_csv(self, dataframe: pd.DataFrame) -> None:
        if not dataframe.empty:
            filename = sg.popup_get_file(
                "Save As",
                save_as=True,
                no_window=True,
                file_types=(("CSV Files", "*.csv"),),
                default_extension=".csv",
            )
            if filename:
                dataframe.to_csv(filename, index=False)
                sg.popup("Export Successful", f"The file was saved as {filename}.")
        else:
            sg.popup("Warning", "There is no data to export.")

    def create_main_window(self) -> sg.Window:
        """
        GUIウィンドウを作成する。
        """
        self.get_misclassified_data()
        cm_data, cm_header_list = self.get_confusion_matrix_lists()

        # 閾値や誤分類タイプ、ユーザーIDを選択する部分のレイアウト
        layout_1 = [
            [
                sg.Text("①Select threshold:", size=(25, 1)),
                sg.Combo(
                    ["{:.2f}".format(v / 100) for v in range(50, 100, 5)],
                    key="threshold",
                    enable_events=True,
                    size=(20, 1),
                ),
            ],
            [
                sg.Text("②Select misclassification type:", size=(25, 1)),
                sg.Combo(
                    ["FP", "FN"], key="misclass_type", enable_events=True, size=(20, 1)
                ),
            ],
            [
                sg.Text("③Select user id:", size=(25, 1)),
                sg.Combo([], key="user_id", enable_events=True, size=(20, 1)),
            ],
            [sg.Button("Display", size=(10, 1)), sg.Button("Close", size=(10, 1))],
            [sg.Button("Plot History", size=(22, 1))],
        ]

        # 混合行列を表示する部分のレイアウト
        layout_2 = [
            [
                sg.Table(
                    values=cm_data,
                    headings=cm_header_list,
                    max_col_width=35,
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification="right",
                    num_rows=min(25, len(cm_data)),
                    key="cm_table",
                    row_height=20,
                    tooltip="This is a table",
                )
            ]
        ]

        # ユーザーの決済履歴の表を表示する部分のレイアウト
        layout_3 = [
            [
                sg.Table(
                    values=[],
                    headings=[column for column in self.columns],
                    display_row_numbers=False,
                    auto_size_columns=True,
                    key="table",
                    num_rows=20,
                    vertical_scroll_only=False,
                )
            ],
            [
                sg.Button("Export to CSV", size=(15, 1)),
                sg.Button("Open Sub Window", size=(15, 1)),
            ],
        ]

        # layout_4 = [[sg.Canvas(key="-CANVAS-")]]

        column_1 = sg.Column(layout_1, key="Col1")
        column_2 = sg.Column(layout_2, key="Col1")
        column_3 = sg.Column(layout_3, key="Col3")
        # column_4 = sg.Column(layout_4, key="Col4")

        layout = [
            [
                column_1,
                sg.Column([], size=(100)),
                column_2,
                # column_4,
            ],
            [column_3],
        ]

        window = sg.Window(
            "MisClassified Inspector",
            layout,
            resizable=True,
            size=(1200, 600),
            finalize=True,
        )

        # # グラフを描画し、キャンバスに表示
        # fig = self.plot_pr_auc()
        # figure_canvas_agg = FigureCanvasTkAgg(fig, window["-CANVAS-"].TKCanvas)
        # figure_canvas_agg.draw()
        # figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)

        return window

    def open_sub_window(self, table_data, columns, row_colors):
        """
        サブウィンドウを開くための関数。
        Args:
            table_data(list): メインウィンドウから渡されるテーブルデータ
            columns(list): テーブルのカラムヘッダー
            row_colors(list): 各行の背景色
        Returns:
            生成されたサブウィンドウオブジェクト
        """
        layout_sub = [
            [
                sg.Table(
                    values=table_data,
                    headings=columns,
                    display_row_numbers=False,
                    auto_size_columns=True,
                    key="table_sub",
                    num_rows=20,
                    vertical_scroll_only=False,
                    row_colors=row_colors,  # 行の背景色を設定
                )
            ],
            [sg.Button("Close Subwindow", size=(15, 1))],
        ]

        window_sub = sg.Window("Sub Window", layout_sub)
        return window_sub

    def get_unique_user_ids(self, data) -> List[str]:
        if self.spark is None:
            return sorted([str(uid) for uid in data[self.user_id_col].unique()])
        else:
            unique_user_ids = data.select(self.user_id_col).distinct().collect()
            return sorted([str(row[self.user_id_col]) for row in unique_user_ids])

    def run(self) -> None:
        """
        GUIを起動し、イベントループを管理する。
        """

        self.get_misclassified_data()
        fp_data = self.get_misclassified_data_by_type("FP")
        fn_data = self.get_misclassified_data_by_type("FN")

        fp_user_ids = self.get_unique_user_ids(fp_data)
        fn_user_ids = self.get_unique_user_ids(fn_data)

        classification_type_index = self.columns.index("classification_type")

        window = self.create_main_window()
        window_sub = None

        while True:
            event, values = window.read(timeout=100)
            if event in (sg.WIN_CLOSED, "Close"):
                break

            if event == "threshold" and values["threshold"]:
                self.threshold = float(values["threshold"])
                self.get_misclassified_data()
                fp_data = self.get_misclassified_data_by_type("FP")
                fn_data = self.get_misclassified_data_by_type("FN")
                fp_user_ids = self.get_unique_user_ids(fp_data)
                fn_user_ids = self.get_unique_user_ids(fn_data)

                cm_data, _ = self.get_confusion_matrix_lists()
                window["cm_table"].update(values=cm_data)

            if event == "misclass_type" and values["misclass_type"]:
                user_ids = (
                    fp_user_ids if values["misclass_type"] == "FP" else fn_user_ids
                )
                window["user_id"].update(values=user_ids)

            if event == "Display" and values["user_id"]:
                self.user_data = self.get_selected_user_data(
                    values["user_id"], self.dataset
                )
                if not self.user_data.empty:
                    row_colors = [
                        (index, "red")
                        if row[classification_type_index] == "FN"
                        else (index, "blue")
                        if row[classification_type_index] == "FP"
                        else (index, None)
                        for index, row in enumerate(self.user_data.values.tolist())
                    ]

                    window["table"].update(
                        values=self.user_data.values.tolist(), row_colors=row_colors
                    )
                else:
                    sg.popup("データが見つかりません。")

            if event == "Plot History":
                selected_user_id = values["user_id"]
                selected_user_data = self.get_selected_user_data(
                    selected_user_id, self.dataset
                )

                plot_payment_history(
                    user_data=selected_user_data,
                    datetime_col=self.datetime_col,
                    price_col=self.price_col,
                    label_col=self.label_col,
                )

            if event == "Export to CSV":
                try:
                    self.export_to_csv(self.user_data)
                except AttributeError:
                    sg.popup("データが表示されていません。")

            if event == "Open Sub Window":
                # サブウィンドウを開く
                window_sub = self.open_sub_window(
                    self.user_data.values.tolist(), self.columns, row_colors
                )

            if window_sub:
                event_sub, values_sub = window_sub.read(timeout=100)
                if event_sub in (sg.WIN_CLOSED, "Close Subwindow"):
                    window_sub.close()
                    window_sub = None

        window.close()
