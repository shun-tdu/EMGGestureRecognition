import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from MyDataSet import DataManager, MyDataset
from MyClass.NN.TrainUtils import TrainUtil

from Config import Config
from Models.ModelArchitecture.CnnBaseGestureRecognizer import CnnBaseRecognizer

if __name__ == '__main__':
    config = Config(
        fs=200,
        num_channel=8,
        window_size=150,
        stride=30,
        train_ratio=0.7,
        validation_ratio=0.3,
        batch_size=50,
        num_epoch=300,
        learning_rate=1e-4,
        data_path='./15Subjects-7Gestures/',
        subject_label=('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'),
        data_header=('emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8'),
        gesture_label=('fistdwn', 'neut', 'left', 'right'),
    )

    model_path = 'Models/TrainedModel/CnnBaseGestureRecognizer.pth'

    # デバイス設定（GPUが利用可能ならGPUを使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習データの生成
    data_manager = DataManager(config=config)

    train_data_set = MyDataset(*data_manager.train_data)
    val_data_set = MyDataset(*data_manager.val_data)

    train_data_loader = DataLoader(train_data_set, batch_size=config.batch_size)
    val_data_loader = DataLoader(val_data_set, batch_size=config.batch_size)

    model = CnnBaseRecognizer(input_channels=8,
                              seq_length=config.window_size,
                              num_classes=len(config.gesture_label)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学習・検証ループ
    train_losses = []
    val_losses = []

    train_util = TrainUtil()

    for epoch in range(config.num_epoch):
        print(f"Epoch[{epoch + 1}/{config.num_epoch}]")
        train_loss = train_util.train(model, train_data_loader, criterion, optimizer, device)
        val_loss = train_util.evaluate(model, val_data_loader, criterion, device, mode="Validation")
        print(f"  Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # モデルを保存
    torch.save(model.state_dict(), model_path)

    # 学習が終了したら、損失の推移をプロットする
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.num_epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config.num_epoch + 1), val_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Eval Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
