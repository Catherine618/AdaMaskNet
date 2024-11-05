import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv2d_shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d_shortcut, self).__init__()  # 添加这行代码

        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size= kernel_size,
                                  stride = stride,
                                  padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )

        self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size= (1,3),
                              stride = (1,1),
                              padding = (0,1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.BatchNorm2d(out_channels),
        )


    def forward(self, X):
        identity = self.shortcut(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = X + identity
        result = F.relu(X)
        return result


class ResCNN(nn.Module):
    def __init__(self, dataset, num_classes, input_shape, largeKernal = False):
        super(ResCNN, self).__init__()

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

        self.conv1_block = Conv2d_shortcut(in_channels=1, out_channels=out_channels_1, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2_block = Conv2d_shortcut(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv3_block = Conv2d_shortcut(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size = kernel_size, stride = stride, padding = padding)


        self.flatten_size = self._get_flatten_size(input_shape)
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def _get_flatten_size(self, input_shape):
        try:
            with torch.no_grad():
                    dummy_input = torch.zeros(1, 1, *input_shape)  # 调整输入张量的形状
                    dummy_output = self.conv1_block(dummy_input)
                    dummy_output = self.conv2_block(dummy_output)
                    dummy_output = self.conv3_block(dummy_output)
                    flatten_size = dummy_output.numel()
                    print(f"Flatten size: {flatten_size}")  # 添加这行以打印扁平化大小

            return flatten_size
        except Exception as e:
            print("Error in calculating flatten size:", e)


    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.fc(x)
        return x, features


