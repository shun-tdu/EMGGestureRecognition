from typing import Callable, List, Tuple, Dict, Any

import torch
from torch import Tensor
from torch.nn.functional import conv1d
from scipy.signal import butter, filtfilt


class SignalProcessorTorch:
    """
    PyTorchによる信号処理クラス

    add_stepに関数と引数を指定して使う

    ↓ Usage

    processor.add_step(SignalProcessor.rectification)
    processor.add_step(SignalProcessor.smooth_filter, window_size=200)
    processor.add_step(SignalProcessor.normalize_signal)
    """

    def __init__(self, device: str = 'cpu'):
        # 実行する処理をリストで管理
        self.processing_steps: List[Tuple[Callable[..., Any], Dict[str, Any]]] = []
        self.device = device

    def add_step(self, step_function: Callable[..., Any], **kwargs: Any):
        """
        信号処理ステップを追加

        :param step_function: 信号処理の関数
        :param kwargs: 信号処理関数に渡す追加の引数
        """
        self.processing_steps.append((step_function, kwargs))

    def process(self, signal: Tensor) -> Tensor:
        """
        信号処理を実行

        :param signal: 入力信号 [ch, buffer_size] の ndarray
        :return: 処理後の信号 [ch, buffer_size] の ndarray
        """
        processed_signal = signal.to(self.device)
        # processed_signal = processed_signal.t()
        for step_function, kwargs in self.processing_steps:
            processed_signal = step_function(processed_signal, **kwargs)

        return processed_signal

    @staticmethod
    def bandpass_filter(signal: Tensor, low_cut: float, high_cut: float, fs: float, order: int) -> Tensor:
        nyquist = 0.5 * fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = butter(order, [low, high], btype='band')
        b = torch.tensor(b, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        filtered_signal = filtfilt(b.cpu().numpy(), a.cpu().numpy(), signal.cpu().numpy(), axis=-1)
        filtered_signal = filtered_signal.copy()
        return torch.tensor(filtered_signal, device=signal.device)

    @staticmethod
    def rectification(signal: Tensor) -> Tensor:
        return torch.abs(signal)

    @staticmethod
    def smooth_filter(signal: Tensor, window_size: int) -> Tensor:
        kernel = torch.ones(1, 1, window_size, device=signal.device) / window_size
        signal = signal.unsqueeze(1)
        smoothed_signal = conv1d(signal, kernel, padding=window_size // 2)
        return smoothed_signal.squeeze(1)

    @staticmethod
    def normalize_signal(signal: Tensor) -> Tensor:
        max_values = torch.max(signal, dim=-1, keepdim=True).values
        return signal / max_values
