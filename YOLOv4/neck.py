## Implement Path Aggregation Network 

import torch
import torch.nn as nn
from typing import List
from backbone import BaseBlock, DarkNetBottleneck, CSPStage, DarkNet53


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
    def __init__(self, in_channels_list, num_achor, num_classes) :
        super().__init__()
        '''
        backbone : list of 3
        top-down :
        bottom-up :
        '''
        final_channels = num_anchor * (5 + num_classes)
        # backbone : [(cbl_3, SPP, clb_3), cbl, cbl] 
        # backbone output : [b1, b2, b3]
        self.backbone = DarkNet53()
        
        # [256, 512, 1024]
        x1_channel, x2_channel, x3_channel = in_channels_list


        self.backbone_func1 = BaseBlock(in_channels = x1_channel, out_channels = x1_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# 256x76x76
        self.backbone_func2 = BaseBlock(in_channels = x2_channel, out_channels = x2_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# 512x38x38
        # CBL_3, SPP, CBL_3
        self.backbone_func3 =  nn.Sequential(*[BaseBlock(x3_channel, x3_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                                SPP(), # SPP expand channels 4 times larger.
                                                BaseBlock(x3_channel * 4, x3_channel)
                                            )# 1024x19x19

        self.path_func1 = nn.Sequential(*[BaseBlock(x1_channel, x1_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                          nn.Upsample(scale_factor= 2, mode = 'bilinear'),
                                            BaseBlock(x1_channel, x1_channel)
                                            )# CBL_5, UP, CBL
                        
        # CBL_5, UP, CBL
        self.path_func2 = nn.Sequential(*[BaseBlock(x2_channel, x2_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                          nn.Upsample(scale_factor= 2, mode = 'bilinear'),
                                            BaseBlock(x2_channel, x2_channel)
                                            )# CBL_5, UP, CBL
        
        self.head_func1 = BaseBlock(in_channels = x1_channel, out_channels = x1_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL
        self.head_func2_1 = nn.Sequential(*[BaseBlock(x2_channel, x2_channel, act_fn = 'leakyrelu') for _ in range(5)])# CBL_5
        self.head_func2_2 = BaseBlock(in_channels = x2_channel, out_channels = x2_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.head_func3 = nn.Sequential(*[BaseBlock(x3_channel, x3_channel, act_fn = 'leakyrelu') for _ in range(5)])# CBL_5

        self.result_func1 = BaseBlock(in_channels = x1_channel, out_channels = final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.result_func2 = BaseBlock(in_channels = x2_channel, out_channels = final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.result_func3 = BaseBlock(in_channels = x2_channel, out_channels = final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL


    # top-down : [(cbl_5, up, cbl), (cbl_5, up, cbl)]
    def top_down(self, b1, b2, b3, func1, func2) :
        '''
        b represents backbone output.
        b1 has the largest feature size and b3 has the smallest.
        func{num} corresponds to b{num}. 
        '''
        b3 = b3
        b2 = func1(b2 + b3)
        b1 = func2(b2 + b1)

        return (b1, b2, b3)

    def bottom_up(self, x1, x2, x3, func1, func2_1, func2_2, func3) :
        '''
        x1 has the largest feature size and x3 has the smallest.
        func{num} corresponds to x{num}. 
        func2 is splitted to func2_1 and func2_2 as the diagram
        '''
        x1 = x1
        x2 = func2_1(func3(x1) + x2)
        x3 = func1(func2_2(x2) + x3)

        return (x3, x2, x1)


    def forward(self, x) :
        b1, b2, b3 = self.backbone(x)
        b1 = self.backbone_func1(b1)
        b2 = self.backbone_func2(b2)
        b3 = self.backbone_func3(b3)
        # top down
        p1, p2, p3 = self.top_down(b1, b2, b3, self.path_func1, self.path_func2)
        # bottom up
        n1, n2, n3 = self.bottom_up(p1, p2, p3, self.head_func1, self.head_func2_1, self.head_func2_2, self.head_func3)

        output1 = self.result_func1(n1)
        output2 = self.result_func1(n2)
        output3 = self.result_func1(n3)

        return (output1, output2, output3)