'''implement DarkNet53'''

import torch
import torch.nn as nn


class BaseBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, act_fn = 'mish') :
        super().__init__()
        if act_fn.lower() == 'mish' :
            activation = nn.Mish()
        elif act_fn.lower() == 'leakyrelu' :
            activation = nn.LeakyReLU()
        elif act_fn.lower() == 'relu' :
            activation = nn.ReLU()
        else :
            raise ValueError(f'{act_fn} activation function is not covered. add on DarkNetBottleneck module.')

        self.activation = activation
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        self.activation)
                        
    def forward(self, x) :
        return self.conv(x)
        

class DarkNetBottleneck(nn.Module) :
    def __init__(self, in_channels, out_channels, expansion = 2, act_fn = 'mish') :
        super().__init__()

        mid_channels = int(out_channels / expansion)
        self.conv1 = BaseBlock(in_channels, mid_channels, act_fn = act_fn, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = BaseBlock(mid_channels, out_channels, act_fn = act_fn, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x) :
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output += residual

        return output


class CSPStage(nn.Module) : 
    def __init__(self, in_channels, mid_channels, out_channels, block_fn, expansion, act_fn, num_blocks) :
        '''
        input x will be channel-wise splited into part1 and part2.
        During CSP, only part2 will go through block_fn.
        downsampling layer before CSP, and transition after concatenation.
        input x : (B, C, H, W)
        part1&part2 : (B, C // 2, H, W)
        C should be divisible by 2.

        in_channels : input channel of downsample layer. downsample layer reduce the feature size by 2
        mid_channels : output channel of downsample layer and input channel of cspblock
        block_fn : block function that will be applied on part2. For Darknet53, we are going to use DarkNetBottleneck.
        expansion : expansion of block_fn. e.g) expansion=2, C_in -> C_out//2 -> C_out. C means channel.
        act_fn : activation function. For DarkNet53, we are going to use mish.
        num_blocks : number of iterations of block_fn
        '''
        super().__init__()
        self.downsample = BaseBlock(in_channels, mid_channels, kernel_size = 3, stride = 2, padding = 1)
        self.cspblock = nn.Sequential()

        block_channels = mid_channels // 2 # input_channel for part2

        for i in range(num_blocks) :

            block = block_fn(in_channels = block_channels, 
                                             out_channels = block_channels,
                                             expansion = expansion,
                                             act_fn = act_fn
                                             ) # this only covers DarkNetBottleneck module.

            self.cspblock.add_module(f'partial_block_{i+1}', block )
            
        self.after_cspblock = BaseBlock(in_channels = block_channels, 
                                        out_channels = block_channels,
                                        kernel_size = 1,
                                        stride = 1,
                                        padding = 0,
                                        )
        
        self.transition = BaseBlock(in_channels = 2 * block_channels, 
                                        out_channels = out_channels,
                                        kernel_size = 1,
                                        stride = 1,
                                        padding = 0,
                                        )


    def forward(self, x) :
        x = self.downsample(x)
        split = x.shape[1] // 2
        part1, part2 = x[:, :split], x[:, split:]

        part2 = self.cspblock(part2)
        part2 = self.after_cspblock(part2).contiguous()

        output = torch.cat([part1, part2], dim = 1)
        output = self.transition(output)

        return output


class DarkNet53(nn.Module) :
    '''
    initial layer : conv(3,3,32)/1, mish

    in_channels  : [3, 32,  64,  64, 128,  256]
    mid_channels : [64, 128, 256, 512, 1024]
    out_channels : [64,  64, 128, 256,  512]

    num_blocks of cspstage : [1,2,8,8,4]
    '''
    def __init__(self, act_fn, block_fn, expansion, in_channels_list = [], mid_channels_list = [], out_channels_list = [], num_blocks_list = []) :
        super().__init__()
        
        self.input_layer = BaseBlock(in_channels_list[0], in_channels_list[1], kernel_size = 3, stride = 1, padding = 1)
        
        self.modulelist = nn.Sequential()
        for i, num_blocks in enumerate(num_blocks_list) :
            
            cspstage = CSPStage(in_channels = in_channels_list[i+1], 
                            mid_channels = mid_channels_list[i], 
                            out_channels = out_channels_list[i], 
                            block_fn = block_fn, 
                            expansion = expansion, 
                            act_fn = act_fn, num_blocks = num_blocks)
            self.modulelist.add_module(f'CSPStage_{i+1}', cspstage)

    def forward(self, x) :
        output = self.input_layer(x)
        for stage in self.modulelist :
            output = stage(output)
        return output

    

if __name__ == '__main__' :
    in_channels_list  = [3, 32,  64,  64, 128,  256]
    mid_channels_list = [64, 128, 256, 512, 1024]
    out_channels_list = [64,  64, 128, 256,  512]
    num_blocks_list   = [1,2,8,8,4]

    model = DarkNet53(act_fn = 'mish', block_fn = DarkNetBottleneck, expansion = 2, 
                    in_channels_list = in_channels_list,
                    mid_channels_list = mid_channels_list,
                    out_channels_list = out_channels_list,
                    num_blocks_list = num_blocks_list
                    )

    model(torch.randn((1,3,512, 512))).shape