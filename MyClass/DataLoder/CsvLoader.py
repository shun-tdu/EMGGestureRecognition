from typing import overload, Union

import numpy as np
import pandas
import pandas as pd


class CsvLoader:
    def __init__(self, data_dir_path: str, sampling_frequency: float, header: tuple):
        """
        CSVデータの読み込み，書き込みを担うヘルパークラス

        :param data_dir_path (float):読み込むCSVファイルのパス
        :param sampling_frequency (float): サンプリング周波数 [1/s]
        :param header (tuple): 読み込みたいヘッダの名前リスト
        """
        self._data_path = data_dir_path
        self._fs = sampling_frequency
        self._header = header

        self._load_data()

    def _load_data(self):
        """
        指定した列名のリストを使ってデータフレームから該当列を抜き出す。
        """
        data_frame = pd.read_csv(self._data_path)
        valid_columns = [col for col in self._header if col in data_frame.columns]
        if not valid_columns:
            raise ValueError("指定したヘッダーがCSVファイルに存在しません．")
        self._extracted_data = data_frame[valid_columns]

    def get_data_sequentially(self, start_time: float, end_time: float, header: tuple) -> np.array:
        """
        開始時刻と終了時刻の範囲のデータを1行ずつ返すジェネレータ

        :param start_time :(float) 読み込みの開始時刻
        :param end_time :(float) 読み込みの終了時刻
        :param header :(tuple) 読み込むデータのヘッダ
        :return: np.array[, row_size]
        """

        # 指定されたヘッダーがデータに存在するかチェック
        valid_columns = [col for col in header if col in self._extracted_data.columns]
        if not valid_columns:
            raise ValueError("指定したヘッダーが存在しません．")

        # 時間範囲をサンプル数に変換してデータをスライス
        start_idx = int(start_time * self._fs)
        end_idx = int(end_time * self._fs)

        # 範囲外チェック
        if start_idx < 0 or end_idx > len(self._extracted_data):
            raise ValueError("指定した時間範囲がデータの範囲外です．")

        for index in range(start_idx, end_idx):
            yield self._extracted_data[valid_columns].to_numpy()[index]

    @overload
    def get_data(self, start_time: int, end_time: int, header: tuple) -> np.ndarray:
        ...

    @overload
    def get_data(self, start_time: float, end_time: float, header: tuple) -> np.ndarray:
        ...

    def get_data(self, start_time: Union[int, float], end_time: Union[int, float], header: tuple) -> np.ndarray:
        """
        任意の時間のデータを読み出す

        :param start_time :(float) 読み込みの開始時刻
        :param end_time :(float) 読み込みの終了時刻
        :param header :(tuple) 読み込むデータのヘッダ
        :return (np.ndarray): 指定した時間範囲および列のデータ
        """

        # 指定されたヘッダーがデータに存在するかチェック
        valid_columns = [col for col in header if col in self._extracted_data.columns]
        if not valid_columns:
            raise ValueError("指定したヘッダーが存在しません．")

        # 時間範囲をサンプル数に変換してデータをスライス
        if isinstance(start_time, float) and isinstance(end_time, float):
            start_idx = int(start_time * self._fs)
            end_idx = int(end_time * self._fs)
        elif isinstance(start_time, int) and isinstance(end_time, int):
            start_idx = start_time
            end_idx = end_time
        else:
            raise TypeError("指定した範囲がデータの範囲外です．")

        # 時間範囲がデータ範囲外でないかをチェック
        if start_idx < 0 or end_idx > len(self._extracted_data):
            raise ValueError("指定した時間範囲がデータの範囲外です．")

        return self._extracted_data[valid_columns].to_numpy()[start_idx:end_idx]

    @property
    def row_size(self) -> int:
        if self._extracted_data.shape[1] is None:
            raise ValueError("The synergy number has not been initialized yet.")
        return self._extracted_data.shape[1]

    @property
    def col_size(self) -> int:
        if self._extracted_data.shape[0] is None:
            raise ValueError("The synergy number has not been initialized yet.")
        return self._extracted_data.shape[0]
