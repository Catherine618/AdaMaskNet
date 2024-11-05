import torch.nn as nn
import torch

from models.adaMaskNet.TemporalRF import TemporalRF
from models.adaMaskNet.structure_build import generate_layer_parameter_list

class AdaMaskNet(nn.Module):
    def __init__(self,
                 n_class,
                 input_shape,
                 temporal_parameter_number_of_layer_list,
                 stride,
                 Max_kernel_size = 89,
                 averagepool_size = (9,1),
                 sc = True):

        super(AdaMaskNet, self).__init__()

        self.n_class = n_class
        self.input_shape = input_shape
        self.temporal_parameter_number_of_layer_list = temporal_parameter_number_of_layer_list
        self.stride = stride
        self.averagepool_size = averagepool_size
        self.Max_kernel_size = Max_kernel_size
        self.sc = sc
        self.layer_list = nn.ModuleList()

        temporal_receptive_field_shape = min(int(input_shape[-1] / 4), self.Max_kernel_size)
        temporal_layer_parameter_list = generate_layer_parameter_list(1,
                                                                      temporal_receptive_field_shape,
                                                                      temporal_parameter_number_of_layer_list,
                                                                      in_channel=1)

        self.temporal_wise = TemporalRF(temporal_layer_parameter_list, self.stride)
        self.averagepool = nn.AdaptiveAvgPool2d(averagepool_size)
        self.flatten_size = self._get_flatten_size(input_shape)
        self.linear = nn.Linear(self.flatten_size, self.n_class)

    def _get_flatten_size(self, input_shape):
        extend_input_shape = [1, input_shape[0], input_shape[1]]
        dummy_input_shape = (1, *extend_input_shape)
        dummy_output = self.temporal_wise(torch.zeros(dummy_input_shape))
        dummy_pooled_output = self.averagepool(dummy_output)
        flattened_size = dummy_pooled_output.view(dummy_pooled_output.size(0), -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.temporal_wise(x)
        features = x
        x = self.averagepool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, features


