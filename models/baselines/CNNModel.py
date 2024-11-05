import torch.nn as nn
import torch
class CNNModel(nn.Module):
    def __init__(self, num_classes, input_shape, kernel_size):
        super(CNNModel, self).__init__()

        out_channels_1 = 64
        out_channels_2 = 128
        out_channels_3 = 256
        conv1 = nn.Conv2d(1, out_channels_1, (1,7), (1,1), (0, 1))
        conv2 = nn.Conv2d(out_channels_1, out_channels_2, (1,7), (1,1), (0, 1))
        conv3 = nn.Conv2d(out_channels_2, out_channels_3, (1,7), (1,1), (0, 1))



        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(True),

            conv2,
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(True),

            conv3,
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(True),
        )
        self.flatten_size = self._get_flatten_size(input_shape)
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def _get_flatten_size(self, input_shape):
        # 创建一个临时输入张量，计算经过卷积模块后的扁平化大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)  # 调整输入张量的形状
            dummy_output = self.conv_module(dummy_input)
            flatten_size = dummy_output.numel()
        return flatten_size

    def forward(self, x):
        x = self.conv_module(x)
        features = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, features
