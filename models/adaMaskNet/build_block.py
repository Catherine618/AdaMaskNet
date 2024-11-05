import numpy as np
import torch
import torch.nn as nn
import math

def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel, 1, largest_kernel_lenght))
    mask[:,:,:,0:ind_left]=0
    mask[:,:,:,ind_right:]=0
    return mask

def creat_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    for i in layer_parameter_list:
        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)
    mask = np.concatenate(mask_list, axis=0)
    return mask.astype(np.float32)

class build_block(nn.Module):
    def __init__(self, layer_parameters, stride, layindex):
        super(build_block, self).__init__()

        self.layindex = layindex
        self.stride = stride

        os_mask = creat_layer_mask(layer_parameters)
        self.in_channels = os_mask.shape[1]
        self.out_channels = os_mask.shape[0]
        self.max_kernel_size = os_mask.shape[-1]
        self.kernel_size = (1, self.max_kernel_size)
        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)
        self.padding = nn.ConstantPad2d((int((self.max_kernel_size - 1) / 2), int(self.max_kernel_size / 2), 0, 0),
                                        value=0)

        self.rbr_identity = nn.BatchNorm2d(
            num_features=self.in_channels) if self.out_channels == self.in_channels else None
        self.rbr_dense = self.conv_bn_dense()
        self.rbr_1x1 = self.conv_bn_1x1()
        self.nonlinearity = nn.ReLU()


    def conv_bn_dense(self):
        conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                           kernel_size=self.kernel_size, stride=self.stride, bias=True)
        bn = nn.BatchNorm2d(self.out_channels)

        result = nn.Sequential()
        result.add_module('conv', conv2d)
        result.add_module('bn', bn)
        return result

    def conv_bn_1x1(self):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                            kernel_size=(1,1), stride=self.stride, padding=(0,0), bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return result



    def forward(self, X):
        self.rbr_dense.conv.weight.data = self.rbr_dense.conv.weight * self.weight_mask
        result_1 = self.padding(X)
        result_1 = self.rbr_dense(result_1)

        if self.sc:
            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(X)
            result = self.nonlinearity(result_1 + self.rbr_1x1(X) + id_out)
        else:
            result = self.nonlinearity(result_1)

        return result