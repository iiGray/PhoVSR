import torch
from torch import nn

import torch.nn.utils.rnn as rnn_utils
from einops import rearrange
from einops.layers.torch import Rearrange


if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())
    
from model.frontend.resnet import *


class BatchNorm3d(nn.BatchNorm3d):
    def __init__(self,
                 num_features=3,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 *args,**kwargs):
        super().__init__(num_features,
                         eps,
                         momentum,
                         affine,
                         track_running_stats,
                         *args,**kwargs)

class FrontEnd3D(nn.Sequential):
    def __init__(self,grayMode=True):
        in_channels=1 if grayMode else 3
        super().__init__(
            nn.Conv3d(in_channels, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            BatchNorm3d(64),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
#             BatchNorm3d(64)
        )
        
class FrontEndModule(nn.Module):
    def __init__(self,odims,grayMode=True,resnet_type="resnet18"):
        super().__init__()
        self.front3D=FrontEnd3D(grayMode)
        if "36" in resnet_type:
            self.resnet=ResNet([3,4,6,3])
        else:
            self.resnet=ResNet([2,2,2,2])

        self.fc=nn.Linear(512,odims)
        
    def forward(self,x, xl):
        b,t,c,h,w=x.shape
        x=rearrange(x,"b t c h w -> b c t h w")
        x=self.front3D(x)
        x=rearrange(x,"b c t h w -> (b t) c h w")
        x=self.resnet(x)
        
        x=self.fc(x)

        return rearrange(x,"(b t) f -> b t f",b=b)

        
        x=rearrange(x,"b t c h w -> b c t h w")
        x=self.front3D(x)
        x=rearrange(x,"b c t h w -> b t c h w")
        
        
        # (b, t, c, h, w)  ->  (bt, c, h, w)
        packed_x=rnn_utils.pack_padded_sequence(x.transpose(0,1),xl.cpu(),enforce_sorted=False)

        x=self.resnet(packed_x.data)

        #(bt, f)
        x=self.fc(x)

        packed_x =rnn_utils.PackedSequence(data = x,
                                          batch_sizes = packed_x.batch_sizes,
                                          sorted_indices = packed_x.sorted_indices
                                          )
        padded_x, pxl= rnn_utils.pad_packed_sequence(packed_x)
        return padded_x.transpose(0,1)
