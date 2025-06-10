import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class Config:
    """
    筋電位信号（EMG）のジェスチャー分類モデルに必要な設定を保持するためのデータクラス．
    各フィールドは，フィルタ処理や信号解析に関わるパラメータを定義．

    Attributes:
        fs (float):
            サンプリング周波数 [Hz]
        num_channel (int):
            EMG信号のチャンネル数
        window_size (int):
            バッファサイズ
        stride(int):
            ウィンドウを移動させるサイズ
        train_ratio (float):
            Trainデータの比率
        validation_ratio (float):
            Validationデータの比率
        batch_size (int):
            学習のバッチサイズ
        num_epoch (int):
            エポック数
        learning_rate (float):
            学習率
        data_path (str):
            データのパス
        data_header (Tuple[str, ...]):
            読み込むデータのヘッダ
        gesture_label (Tuple[str, ...]):
            識別するジェスチャのラベル
        subject_label (Tuple[str, ...]):
            読み込むSubjectのラベル
    """

    # サンプリング周波数
    fs: float
    # EMGのチャンネル数
    num_channel: int
    # NNに入力する窓長
    window_size: int
    # ウィンドウを移動させるサイズ
    stride: int
    # Trainデータの比率
    train_ratio: float
    # Validationデータの比率
    validation_ratio: float
    # 学習のバッチサイズ
    batch_size: int
    # エポック数
    num_epoch: int
    # 学習率
    learning_rate: float
    # データのパス
    data_path: str
    # 読み込むデータのヘッダ
    data_header: Tuple[str, ...]
    # 識別するジェスチャのラベル
    gesture_label: Tuple[str, ...]
    # 読み込むSubjectのラベル
    subject_label: Tuple[str, ...]
