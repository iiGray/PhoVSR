
from torch import nn
from typing import Union

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())
    
from model.backend.normalize import *
    
class PosWiseFFN(nn.Sequential):
    def __init__(self,idims,hdims,dropout,Norm:Union[PreNorm,AddNorm],scale=1.):
        super().__init__(
            Norm(
                idims,
                nn.Sequential(
                    nn.Linear(idims,hdims),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout),
                    nn.Linear(hdims,idims)
                ),
                scale,
                dropout
            )
        )