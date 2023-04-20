## Implement Path Aggregation Network 

import torch
import torch.nn as nn
from typing import List


class SPP(nn.Module) :
    def __init__(self, scales: List = [1, 5, 9 , 13]) :
        super().__init__()
        '''
        scales : list of kernel size for maxpooling. stride is fixed 1. padding is scale // 2 to fix the output shape.
        output : input_channel x len(scales)
        '''

        self.pool = nn.Sequential()
        for s in scales :
            self.pool.add_module(f'maxpool_{s}x{s}', nn.MaxPool2d(kernel_size = s, stride = 1, padding = s // 2))
        
    def forward(self, x) :
        result = []
        for pool in self.pool :
            result.append(pool(x))

        return torch.cat(result, dim = 1)
