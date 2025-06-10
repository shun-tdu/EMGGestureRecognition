import torch
from torch.utils.data import DataLoader
import numpy as np
from MyDataSet import DataManager, MyDataset
from Config import Config
from Models.ModelArchitecture.CnnBaseGestureRecognizer import CnnBaseRecognizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

if __name__ == '__main__':
    # Config の設定（必要なパラメータを設定）
    config = Config(
        fs=200,
        num_channel=8,
        window_size=150,
        stride=30,
        train_ratio=0.7,
        validation_ratio=0.3,  # ※今回は検証データをテストデータとして利用
        batch_size=50,
        num_epoch=100,
        learning_rate=1e-4,
        data_path='./15Subjects-7Gestures/',
        subject_label=('S3', 'S4'),
        data_header=('emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8'),
        gesture_label=('fistdwn', 'neut', 'left', 'right'),
    )

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataManager を利用してデータを読み込み
    data_manager = DataManager(config=config)
    # ここでは検証データをテストデータとして利用
    test_dataset = MyDataset(*data_manager.train_data)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # モデルの定義と読み込み（保存したパラメータを読み込む）
    model = CnnBaseRecognizer(input_channels=8,
                              seq_length=config.window_size,
                              num_classes=len(config.gesture_label)).to(device)
    model_path = 'Models/TrainedModel/CnnBaseGestureRecognizer.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 推論モードに変更

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    # テストデータに対する推論ループ
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 入力の次元が (batch, window_size, num_channels) の場合、
            # Conv1d 用に (batch, num_channels, window_size) に変換
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # モデルの出力（生のロジット）
            _, predicted = torch.max(outputs, dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 混同行列や分類レポートの作成（オプション）
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.gesture_label,
                yticklabels=config.gesture_label)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=config.gesture_label))
