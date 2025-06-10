from typing import Callable, List, Tuple, Dict, Any
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d


class SignalProcessor:
    """
    信号処理をするクラス

    add_stepに関数と引数を指定して使う

    ↓ Usage

    processor.add_step(SignalProcessor.rectification)
    processor.add_step(SignalProcessor.smooth_filter, window_size=200)
    processor.add_step(SignalProcessor.normalize_signal)
    """

    def __init__(self):
        # 実行する処理をリストで管理
        self.processing_steps: List[Tuple[Callable[..., Any], Dict[str, Any]]] = []

    def add_step(self, step_function: Callable[..., Any], **kwargs: Any) -> None:
        """
        信号処理ステップを追加

        :param step_function: 信号処理の関数
        :param kwargs: 信号処理関数に渡す追加の引数
        """
        self.processing_steps.append((step_function, kwargs))

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        信号処理を実行

        :param signal: 入力信号 [sample, ch] の ndarray
        :return: 処理後の信号 [sample, ch] の ndarray
        """

        processed_signal = signal
        for step_function, kwargs in self.processing_steps:
            processed_signal = step_function(processed_signal, **kwargs)

        return processed_signal

    @staticmethod
    def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int) -> np.ndarray:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal, axis=0)

        return filtered_signal

    @staticmethod
    def rectification(signal: np.ndarray) -> np.ndarray:
        return np.abs(signal)

    @staticmethod
    def smooth_filter(signal: np.ndarray, window_size: int) -> np.ndarray:
        smoothed_emg = np.zeros_like(signal)

        for ch in range(signal.shape[1]):
            smoothed_emg[:, ch] = np.convolve(signal[:, ch], np.ones(window_size) / window_size, mode='same')

        return smoothed_emg

    @staticmethod
    def normalize_signal(signal: np.ndarray) -> np.ndarray:
        min_value = np.min(signal)
        max_value = np.max(signal)
        normalized_signal = (signal - min_value) / (max_value - min_value)

        return normalized_signal

        # チャンネル毎に正規化
        # normalized_signal = np.zeros_like(signal)
        #
        # for ch in range(signal.shape[1]):
        #     mvc_value = np.max(signal[:, ch])
        #     normalized_signal[:, ch] = signal[:, ch] / mvc_value
        #
        # return normalized_signal

    @staticmethod
    def integrate_signal(signal: np.ndarray, dt: float) -> np.ndarray:
        """
        信号を積分する関数
        各チャネルごとに時系列方向で数値積分を実施

        :param signal: 入力信号 [sample, ch] の ndarray
        :param dt: サンプリング間隔（デフォルトは 1.0）
        :return: 積分後の信号 [sample, ch] の ndarray
        """
        integrated_signal = np.cumsum(signal, axis=0) * dt
        return integrated_signal

    @staticmethod
    def local_integrate_signal(signal: np.ndarray, window_size: int, dt: float):
        """
        局所的積分（ローカル統合）を計算する関数
        指定したウィンドウサイズで、各チャネルごとに移動ウィンドウ内の積分値（IEMG値に相当）を計算

        :param signal: 入力信号 [sample, ch] の ndarray
        :param window_size: 積分に使用するウィンドウのサイズ（サンプル数）
        :param dt: サンプリング間隔（デフォルトは 1.0）
        :return: 局所積分後の信号 [sample, ch] の ndarray
        """

        integrated_signal = np.zeros_like(signal)
        kernel = np.ones(window_size) * dt

        for ch in range(signal.shape[1]):
            integrated_signal[:, ch] = np.convolve(signal[:, ch], kernel, mode='same')

        return integrated_signal

    @staticmethod
    def normalize_time_axis(signal:np.ndarray, target_length:int):
        """
        時間方向に正規化を行う関数

        :param signal: 入力信号 [sample, ch] の ndarray
        :param target_length: 正規化後のサンプル数
        :return: 時間方向に正規化された信号 [target_length, ch] の ndarray
        """
        original_length = signal.shape[0]
        original_indices = np.linspace(0, 1, original_length)
        target_indices = np.linspace(0, 1, target_length)
        interpolator = interp1d(original_indices, signal, kind='linear', axis=0)
        resampled_data = interpolator(target_indices)

        return resampled_data
