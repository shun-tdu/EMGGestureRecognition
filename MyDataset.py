import numpy as np
import torch
from torch.utils.data import Dataset

from MyClass.DataLoder.CsvLoader import CsvLoader
from MyClass.SignalProcessor.SignalProcessor import SignalProcessor
from Config import Config


class DataManager:
    def __init__(self, config: Config):
        self._config = config
        self._num_subject = len(config.subject_label)
        self._num_gesture = len(config.gesture_label)
        self._resample_length = 2000

        # Signal Processorの初期化
        self._sig_pr = SignalProcessor()
        self._sig_pr.add_step(SignalProcessor.normalize_signal)
        self._sig_pr.add_step(SignalProcessor.normalize_time_axis, target_length=self._resample_length)

        # モデルに入力するデータ
        self._X_features = []
        # gestureラベル
        self._labels = []

        # Rawデータの読み込み
        self._load_data()

        # Numpyに変換
        self._X_features = np.array(self._X_features)
        self._labels = np.array(self._labels)

        # Train Data,Val Dataの生成
        self._train_data, self._val_data = self._split_data()

    def _load_data(self):
        """
        Configを元にデータを読み込む
        """
        for idx_sub, sub in enumerate(self._config.subject_label):
            for idx_gesture, gesture in enumerate(self._config.gesture_label):
                loading_file_path = self._config.data_path + sub + '/' + 'emg-' + gesture + '-' + sub + '.csv'
                loader = CsvLoader(loading_file_path, self._config.fs, self._config.data_header)
                raw_data = loader.get_data(start_time=0,
                                           end_time=loader.col_size,
                                           header=self._config.data_header)
                filtered_data = self._sig_pr.process(raw_data)

                for i in range(0, self._resample_length - self._config.window_size + 1, self._config.stride):
                    windowed = filtered_data[i:i + self._config.window_size, :]

                    self._X_features.append(windowed)
                    self._labels.append(idx_gesture)

    def _split_data(self):
        num_samples = len(self._X_features)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # インデックスをランダムにシャッフル

        train_size = int(num_samples * self._config.train_ratio)
        val_size = num_samples - train_size

        # インデックスをそれぞれに分割
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        train_data = (self._X_features[train_indices], self._labels[train_indices])
        val_data = (self._X_features[val_indices], self._labels[val_indices])

        return train_data, val_data

    @property
    def train_data(self):
        X, Y = self._train_data
        return torch.from_numpy(X).float(), torch.from_numpy(Y).long()

    @property
    def val_data(self):
        X, Y = self._val_data
        return torch.from_numpy(X).float(), torch.from_numpy(Y).long()


class MyDataset(Dataset):
    """
    PyTorch Dataset クラス
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X (np.ndarray): 入力データ
            Y (np.ndarray): 出力データ
        """
        self.X = X  # (num_samples, window_size , num_ch)
        self.Y = Y  # (num_samples, )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
