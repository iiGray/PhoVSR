from torch import nn

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.backend.transformer.attention import *
from model.backend.positionFFN import *

from template.e2e.utils import State,EvalModule
from typing import Union

class DecoderLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.dec_att=norm_type(
            idims,
            MultiHeadAttention(idims,num_heads,dropout),
            scale=1.,
            dropout=dropout,
        )
        self.src_att=norm_type(
            idims,
            MultiHeadAttention(idims,num_heads,dropout),
            scale=1.,
            dropout=dropout,
        )
        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type)

    def forward(self,x,x_mask,y,y_mask,cache=None):
        if cache is None:
            x=self.dec_att(x,x,x,x_mask)
            x=self.src_att(x,y,y,y_mask)
            x=self.pos_ffn(x)
        else:
            q=x[:,[-1],:]
            q=self.dec_att(q,x,x,x_mask)
            q=self.src_att(q,y,y,y_mask)
            q=self.pos_ffn(q)
            x=torch.concat([cache,q],dim=1)
            
        return x


class Decoder(nn.Module,EvalModule):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0.,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.idims=idims
        self.tok_enc=nn.Embedding(char_list_len,idims)
        self.pos_enc=PositionEncoding(idims,dropout)
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        self.decoder=nn.Sequential(*[
            DecoderLayer(idims,
                         hdims,
                         num_heads,
                         dropout,
                         norm_type) for _ in range(num_layers)
        ])
        
        
        self.fc=nn.Linear(idims,char_list_len)
        
    def forward(self,x,x_valid_len,enc_state,enc_valid_len):
        x=self.tok_enc(x)
        x=self.pos_enc(x)
        
        x_mask=get_mask((x.size(1),x.size(1)),x_valid_len,zero_triu=True)
        
        enc_mask=get_mask((x.size(1),enc_state.size(1)),enc_valid_len,zero_triu=False)

        for decoder_layer in self.decoder:
            x=decoder_layer(x,x_mask,enc_state,enc_mask)
            
        x=self.norm(x)
        
        out=self.fc(x)
        
        return out
    
    def initState(self, eState: State, *args, **kwargs):
        batch_size=len(eState)
        self.caches=[torch.zeros((batch_size,0,self.idims),
                                 dtype=eState.dtype,
                                 device=eState.device)\
                    for _ in range(self.num_layers)]
        
        return eState
    

    def selectNext(self, bk_pref, bk_pred):
        '''
        select next cache after topk

        bk_pref:  (batch_beam_size, )  the  last   ids
        bk_pred:  (batch_beam_size, )  the current pinyin ids

        choose prefix from self.caches
        
        '''
        for i in range(len(self.caches)):
            self.caches[i]=State(self.caches[i]).setitem(bk_pref).feats
            
    def scoring(self,eState: State, dState: State):
        '''
        while evaluating:
        self.training = False
        '''
        
        enc_state,enc_valid_len=eState
        x,x_valid_len=dState
        x=self.tok_enc(x)
        x=self.pos_enc(x)
        
        x_mask=None
        enc_mask=get_mask(enc_state.size(1), enc_valid_len, zero_triu=False)
        
        new_caches=[]
        
        for i,decoder_layer in enumerate(self.decoder):
            x=decoder_layer(x,x_mask,enc_state,enc_mask,self.caches[i])
            
            new_caches.append(x)
            
        self.caches=new_caches
        
        out=self.norm(x)
        out=out[:,-1,:]
#         out=self.norm(out)
        return self.fc(out)
        
        
class RelDecoderLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.dec_att=norm_type(
            idims,
            EncoderRelPosMultiHeadAttn(idims,num_heads,dropout),
            dropout
        )

        self.src_att=norm_type(
            idims,
            RelPosMultiHeadAttention(idims,num_heads,dropout),
            dropout
        )

        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type)

    def forward(self,x,x_mask,y,y_mask,Rx,Ry,cache=None):
        if cache is None:
            x=self.dec_att(x,x,x,Rx,x_mask)
            x=self.src_att(x,y,y,Ry,y_mask)
            x=self.pos_ffn(x)
        else:
            current_step=cache.size(1)+1
            q=x[:,[-1],:]
            q=self.dec_att(q,x,x,Ry,x_mask,current_step)
            q=self.src_att(q,y,y,Ry,y_mask,current_step)
            q=self.pos_ffn(q)

            x=torch.concat([cache,x],dim=1)

        return x


class RelDecoder(nn.Module,EvalModule):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.idims=idims
        self.tok_enc=nn.Embedding(char_list_len,idims)
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        self.pos_enc=RelPositionEncoding(idims,dropout)
        self.decoder=nn.Sequential(*[
            RelDecoderLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type
                            ) for _ in range(num_layers)
        ])

        self.fc=nn.Linear(idims,char_list_len)
    def forward(self,x,x_valid_len,enc_state,enc_valid_len):

        x=self.tok_enc(x)
        x,Rx=self.pos_enc(x)

        enc_state,Re=self.pos_enc(enc_state)

        x_mask=get_mask((x.size(1),x.size(1)),x_valid_len,zero_triu=True)
        enc_mask=get_mask((x.size(1),enc_state.size(1)),enc_valid_len,zero_triu=False)

        for decoder_layer in self.decoder:
            x=decoder_layer(x,x_mask,enc_state,enc_mask,Rx,Re)

        x=self.norm(x)

        out=self.fc(x)

        return out
    
    def initState(self, eState: State, *args, **kwargs) -> State:
        batch_size=len(eState)

        self.caches=[torch.zeros((batch_size,0,self.idims),
                                 dtype=eState.dtype,
                                 device=eState.device)]
    
        return eState
    
    def selectNext(self, bk_pref, bk_pred, *args, **kwargs):
        for i in range(len(self.caches)):
            self.caches[i]=State(self.caches[i]).setitem(bk_pref).feats
    
    def scoring(self, eState: State, dState: State, *args, **kwargs):
        enc_state,enc_valid_len=eState
        x,x_valid_len=dState

        x=self.tok_enc(x)
        
        x,Rx=self.pos_enc(x)
        enc_state,Re=self.pos_enc(enc_state)

        x_mask=None

        enc_mask=get_mask(enc_state.size(1),enc_valid_len,zero_triu=False)

        new_caches=[]


        for i,decoder_layer in enumerate(self.decoder):
            x=decoder_layer(x,x_mask,enc_state,enc_mask,Rx,Re,self.caches[i])

            new_caches.append(x)
        
        self.caches=new_caches

        x=self.norm(x)

        return self.fc(x[:,-1,:])
    
if __name__=="__main__":
    print("success")

