import torch.nn as nn
import torch

class HAR_CNN(nn.Module):
    def __init__(self, dataset, num_classes, input_shape, largeKernal = False):
        super(HAR_CNN, self).__init__()

        self.dataset = dataset

        out_channels_1 = 64
        out_channels_2 = 128
        out_channels_3 = 256

        if dataset == 'PAMAP2':
            if largeKernal:
                kernel_size = (1, 41)
                stride = (1, 2)
                padding = (0, 20)
            else:
                kernel_size = (1,3)
                stride = (1,2)
                padding = (0,1)


        elif dataset == 'UCI_HAR':
            if largeKernal:
                kernel_size = (1, 31)
                stride = (1, 3)
                padding = (0, 15)
            else:
                kernel_size = (1, 6)
                stride = (1, 3)
                padding = (0, 1)

        elif dataset == 'UniMiB_SHAR':
            if largeKernal:
                kernel_size = (1, 37)
                stride = (1, 2)
                padding = (0, 18)
            else:
                kernel_size = (1, 6)
                stride = (1, 2)
                padding = (0, 1)

        elif dataset == 'WISDM':
            if largeKernal:
                kernel_size = (1, 47)
                stride = (1, 2)
                padding = (0, 23)
            else:
                kernel_size = (1, 8)
                stride = (1, 2)
                padding = (0, 1)

        conv1 = nn.Conv2d(1, out_channels_1, kernel_size, stride, padding)
        conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride, padding)
        conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size, stride, padding)

        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(True),

            conv2,
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(True),

            conv3,
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(True)
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
        # x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
        features = x
        x = self.fc(x)
        return x, features

