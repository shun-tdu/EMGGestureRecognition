import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnBaseRecognizer(nn.Module):
    def __init__(self, input_channels: int, seq_length: int, num_classes: int):
        """
        Args:
            input_channels (int): 入力信号のチャネル数
            seq_length (int): 入力シーケンスの長さ
            num_classes (int): 分類クラス数
        """
        super(CnnBaseRecognizer, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        def calc_out_length(l, kernel_size, pool_size):
            l = l - kernel_size + 1
            l = l // pool_size
            return l

        # conv1 + pool1
        l1 = calc_out_length(seq_length, kernel_size=3, pool_size=2)
        # conv2 + pool2
        l2 = calc_out_length(l1, kernel_size=3, pool_size=2)

        self.flatten_dim = 64 * l2

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape = (batch_size, input_channels, seq_length)
        Returns:
            出力は softmax を通した確率分布
        """
        # Convolution
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
