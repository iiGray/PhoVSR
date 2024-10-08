from typing import Union

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.backend.roformer.attention import *
from model.backend.positionFFN import *
from model.backend.conv import *

class RoEncoderLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True
                 ):
        super().__init__()
        self.use_conv=use_conv
        
        scale = 0.5 if use_conv else 1
        
        if use_conv:
            self.conv_ffn=PosWiseFFN(idims,hdims,dropout,norm_type,scale=scale)

            self.conv=norm_type(
                idims,
                ConvolutionModule(idims,31),
                scale=1.,
                dropout=dropout,
            )

        self.enc_att=norm_type(
            idims,
            RoMultiHeadAttention(idims,num_heads,dropout),
            dropout
        )

        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type,scale=scale)
        
        if self.use_conv:
            self.conv_norm=nn.LayerNorm(idims,eps=1e-12)
        
    def forward(self, x, mask):
        if self.use_conv:
            x=self.conv_ffn(x)
        
        x=self.enc_att(x, x, x, mask)
        
        if self.use_conv:
            x=self.conv(x)
        
        x=self.pos_ffn(x)
        
        if self.use_conv:
            x=self.conv_norm(x)
            
        return x



class RoEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0.,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.char_list_len=char_list_len
        self.norm_type=norm_type
        if char_list_len is not None:
            self.tok_enc=nn.Embedding(char_list_len,idims)
        
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        
        self.encoder=nn.Sequential(*[
            RoEncoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type,
                            use_conv
                            ) for _ in range(num_layers)
        ])

    def forward(self,x,valid_len):
        if self.char_list_len is not None:
            x=self.tok_enc(x)
        if self.norm_type is AddNorm:
            x=self.norm(x)  
        
        mask=get_mask((x.size(1),x.size(1)),valid_len,zero_triu=False)
        
#         ones=torch.ones((x.size(1),x.size(1)),dtype=mask.dtype,device=mask.device)
#         mask|=( ~(ones.tril(3)^ones.tril(-3)) )[None,None,:,:]
        
        for encoder_layer in self.encoder:
            x=encoder_layer(x,mask)
        
        if self.norm_type is PreNorm:
            x=self.norm(x)  
        
        return x,valid_len
        
