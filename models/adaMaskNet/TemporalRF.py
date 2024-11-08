import torch.nn as nn
from models.adaMaskNet.build_block import build_block

class TemporalRF(nn.Module):
    def __init__(self,layer_parameter_list, stride):
        super(TemporalRF, self).__init__()

        self.layer_parameter_list = layer_parameter_list #每层的参数列表
        self.stride = stride
        self.layer_list = nn.ModuleList() #用于存储layer
        self.output_channels = 0  # 初始化输出通道数为0

        for i in range(len(layer_parameter_list)):# 遍历每层参数
            layer = build_block(layer_parameter_list[i], stride = self.stride,  layindex=i)  # 根据当前参数创建一个卷积层
            self.layer_list.append(layer)

        last_layer_index = max([i for i, flag in enumerate(self.layerflag) if flag]) #找到最后一个标记为True的Layer
        for final_layer_parameters in layer_parameter_list[last_layer_index]:
                self.output_channels = self.output_channels + final_layer_parameters[1]

    def forward(self, x):
        for layer in self.layer_list:
                x = layer(x)
        return x

