from typing import Union

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.backend.transformer.attention import *
from model.backend.positionFFN import *
from model.backend.conv import *
from model.frontend import *

class EncoderLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 use_cache=None,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.use_cache=use_cache
        self.enc_att=norm_type(
            idims,
            MultiHeadAttention(idims,num_heads,dropout),
            scale=1.,
            dropout=dropout,
        )
        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type)
        
        
    def forward(self, x, mask):
        x=self.enc_att(x, x, x, mask)
        x=self.pos_ffn(x)
        return x



class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0.,
                 use_cache=False,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.use_cache=use_cache
        self.char_list_len=char_list_len
        if char_list_len is not None:
            self.tok_enc=nn.Embedding(char_list_len,idims)
        self.pos_enc=PositionEncoding(idims,dropout=dropout)
        self.encoder=nn.Sequential(*[
            EncoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            use_cache,
                            norm_type
                            ) for _ in range(num_layers)
        ])

    def forward(self,x,valid_len):
        if self.char_list_len is not None:
            x=self.tok_enc(x)
        x=self.pos_enc(x)
        mask=get_mask((x.size(1),x.size(1)),valid_len,zero_triu=False) \
            if self.training else None
        for encoder_layer in self.encoder:
            x=encoder_layer(x,mask)

        return x,valid_len
        

class RelEncoderLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True,
                 ):
        super().__init__()
        self.use_conv=use_conv
        
        scale=0.5 if self.use_conv else 1
        
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
            EncoderRelPosMultiHeadAttn(idims,num_heads,dropout),
            scale=1.,
            dropout=dropout,
        )

        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type,scale=scale)
        
        if use_conv:
            self.conv_norm=nn.LayerNorm(idims,eps=1e-12)

         
    def forward(self, x, mask, R=None):
        if self.use_conv:
            x=self.conv_ffn(x)
            
        x=self.enc_att(x, x, x, R, mask)
        
        if self.use_conv:
#             x.masked_fill_(mask[:,0,0,:][:,:,None],0.)
            x=self.conv(x)
        
        x=self.pos_ffn(x)
        
        if self.use_conv:
            x=self.conv_norm(x)
        
        return x



class RelEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.use_conv=use_conv
        self.char_list_len=char_list_len
        self.norm_type=norm_type
        if char_list_len is not None:
            self.tok_enc=nn.Embedding(char_list_len,idims)
        
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        
        self.pos_enc=RelPositionEncoding(idims,dropout=dropout)
        self.encoder=nn.Sequential(*[
            RelEncoderLayer(idims,
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
        x,R=self.pos_enc(x)
        
        if self.norm_type is AddNorm:
            x=self.norm(x)  
        
        mask=get_mask(x.size(1),valid_len,zero_triu=False)
        
        
        for encoder_layer in self.encoder:
            x=encoder_layer(x,mask,R)

        if self.norm_type is PreNorm:
            x=self.norm(x)  
            
        return x,valid_len
    
    
    

class RelEncoderAux(nn.Module):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.use_conv=use_conv
        self.char_list_len=char_list_len
        self.norm_type=norm_type
        
        self.frontEnd=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
        self.frontEnd_aux=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
        self.frontEnd_linear=nn.Linear(idims,idims)
        
        if char_list_len is not None:
            self.tok_enc=nn.Embedding(char_list_len,idims)
        
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        self.norm_aux=nn.LayerNorm(idims,eps=1e-12)
        
        self.pos_enc=RelPositionEncoding(idims,dropout=dropout)
        self.pos_enc_aux=RelPositionEncoding(idims,dropout=dropout)
        
        self.encoder1=nn.Sequential(*[
            RelEncoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type,
                            use_conv
                            ) for _ in range(num_layers//2)
        ])
        
        self.encoder2=nn.Sequential(*[
            RelEncoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type,
                            use_conv
                            ) for _ in range(num_layers//2)
        ])
        
        self.encoder_aux=nn.Sequential(*[
            RelEncoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type,
                            use_conv
                            ) for _ in range(num_layers//2)
        ])
        
        self.L1=nn.L1Loss(reduction="sum")

        
    def forward(self,x,valid_len):
        x_aux=self.frontEnd_aux(x,valid_len)       
        
        x=self.frontEnd(x,valid_len)


        x,R=self.pos_enc(x)
        x_aux,R_aux=self.pos_enc_aux(x_aux)
        
        if self.norm_type is AddNorm:
            x=self.norm(x)  
            x_aux=self.norm_aux(x_aux)
            
        mask=get_mask(x.size(1),valid_len,zero_triu=False)
        
        
        for ec in self.encoder_aux:
            x_aux=ec(x_aux,mask,R_aux)
        for ec in self.encoder1:
            x=ec(x,mask,R)
        
        
        L1Loss=self.L1(self.frontEnd_linear(x),x_aux)
        
        for ec in self.encoder2:
            x=ec(x,mask,R)
        

        if self.norm_type is PreNorm:
            x=self.norm(x)  
            
        return x,valid_len,L1Loss
        

# class SegEncoderLayer(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  num_heads):
#         super().__init__()


if __name__=="__main__":
    pass