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

class PANet(nn.Module) :
    def __init__(self,) :
        super().__init__()
        '''
        backbone : list of 3
        top-down :
        bottom-up :
        '''

        # backbone : [(cbl_3, SPP, clb_3), cbl, cbl] 
        # backbone output : [b1, b2, b3]

        # top-down : [(cbl_5, up, cbl), (cbl_5, up, cbl)]
        def top_down(x1, x2, x3) :
            x1 = x1
            x2 = func1(x1 + x2)
            x3 = func2(x2 + x3)

            return (x1, x2, x3)

        def bottom_up(x1, x2, x3) :
            x3 = x3
            x2 = func2_1(func3(x3) + x2)
            x1 = func1(func2_2(x2) + x1)

            x3 = cbl(x3)
            x2 = cbl(x2)
            x1 = cbl(x1)

            return (x3, x2, x1)

