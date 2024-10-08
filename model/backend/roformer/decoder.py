from torch import nn

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.backend.roformer.attention import *
from model.backend.positionFFN import *
from template.e2e.utils import State,EvalModule
from typing import Union

class RoDecoderLayer(nn.Module):
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
            RoMultiHeadAttention(idims,num_heads,dropout),
            dropout,
        )
        self.src_att=norm_type(
            idims,
            RoMultiHeadAttention(idims,num_heads,dropout),
            dropout
        )
        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type)

    def forward(self,x,x_mask,y,y_mask,cache=None):
        '''
        current_step: the query steps ,used for roformer positional encoding
        cache       : the current layer's output ,shape (batch_size, current_step-1, feature_size)

        cache is None means training, not None means evaluating
        '''
        if cache is None:
            x=self.dec_att(x,x,x,x_mask)
            x=self.src_att(x,y,y,y_mask)
            x=self.pos_ffn(x)
        else:
            current_step=cache.size(1)
            q=x[:,[-1],:]
            q=self.dec_att(q,x,x,x_mask,current_step)
            q=self.src_att(q,y,y,y_mask,current_step)
            q=self.pos_ffn(q)
            x=torch.concat([cache,q],dim=1)

        return x


class RoDecoder(nn.Module,EvalModule):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_list_len=None,
                 dropout=0.,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 ):
        super().__init__()
        self.idims=idims
        self.num_layers=num_layers
        self.norm_type=norm_type
        self.tok_enc=nn.Embedding(char_list_len,idims)
        
        self.norm=nn.LayerNorm(idims,eps=1e-12)
        
        self.decoder=nn.Sequential(*[
            RoDecoderLayer(idims,
                         hdims,
                         num_heads,
                         dropout,
                         norm_type) for _ in range(num_layers)
        ])
        
        self.fc=nn.Linear(idims,char_list_len)
    
    def forward(self,x,x_valid_len,enc_state,enc_valid_len):
        x=self.tok_enc(x)
        if self.norm_type is AddNorm:
            x=self.norm(x)
            
        x_mask=get_mask((x.size(1),x.size(1)),x_valid_len,zero_triu=True)
        
        enc_mask=get_mask((x.size(1),enc_state.size(1)),enc_valid_len,zero_triu=False)

        for decoder_layer in self.decoder:
            x=decoder_layer(x,x_mask,enc_state,enc_mask)
        
        if self.norm_type is PreNorm:
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
        
        x_valid_len only need one ,because all dState has the same length
        '''
        
        enc_state,enc_valid_len=eState
        x,x_valid_len=dState
        x=self.tok_enc(x)
        
        if self.norm_type is AddNorm:
            x=self.norm(x)

        x_mask=None # make sure that the q is the last num steps

        enc_mask=get_mask(enc_state.size(1),enc_valid_len,zero_triu=False)

        new_caches=[]

        for i,decoder_layer in enumerate(self.decoder):
            x=decoder_layer(x,x_mask,enc_state,enc_mask,self.caches[i])

            new_caches.append(x)
        
        self.caches=new_caches

        if self.norm_type is PreNorm:
            x=self.norm(x)
        out=x[:,-1,:]
#         if self.norm_type is PreNorm:
#             out=self.norm(out)

        return self.fc(out)

    

    
    
class PhoneticRoDecoder(nn.Module,EvalModule):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 char_vocab=None,
                 phoneme_vocab=None,
                 dropout=0.,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.num_layers=num_layers
        self.tok_enc=nn.Embedding(len(phoneme_vocab),idims)

        self.decoder=nn.Sequential(*[
            RoDecoderLayer(idims,
                         hdims,
                         num_heads,
                         dropout,
                         norm_type) for _ in range(num_layers)
        ])
        
        self.fc=nn.Linear(idims,len(phoneme_vocab))
        

        self.map=torch.zeros((len(phoneme_vocab),len(char_vocab)),dtype=torch.long)
        self.c2p=torch.zeros((len(char_vocab),),dtype=torch.long)
        from pypinyin import lazy_pinyin
        for c in char_vocab.char_list:
            lpy=lazy_pinyin(c)[0]
            self.map[phoneme_vocab[lpy],char_vocab[c]]=1
            self.c2p[char_vocab[c]]=phoneme_vocab[lpy]
        self.map[-1,-1]=1
        self.c2p[-1]=len(phoneme_vocab)-1
        for i in range(3):
            self.c2p[i]=i
        

    def forward(self,x,x_valid_len,enc_state,enc_valid_len):
        x=self.tok_enc(x)

        if not self.training:
            x_valid_len=x_valid_len[[0]]
        
        q=x if self.training else x[:,[-1],:]

        current_steps=None if self.training else x.size(1)
        
        x_mask=get_mask((q.size(1),x.size(1)),x_valid_len,
                        zero_triu=True if self.training else False) \
            if self.training or x.size(1)>1 else None
        
        enc_mask=get_mask((q.size(1),enc_state.size(1)),enc_valid_len,zero_triu=False)

        for decoder_layer in self.decoder:
            q=decoder_layer((q,x,x),x_mask,enc_state,enc_mask,current_steps)

        out=self.fc(q)
        
        return out
    
    def initState(self, eState: State):

        self.map=self.map.to(eState.device)
        self.c2p=self.c2p.to(eState.device)
        
        return eState
    
    def scoring(self,eState: State, dState: State):
        '''
        while evaluating:
        self.training = False
        '''
        x,xl=dState
        p=self.c2p[x]
        pState=State(p,xl)
        
        phoneticP=torch.log_softmax(self.forward(*pState,*eState)[:,-1,:],
                                    dim=-1)
        finalP=(phoneticP[:,:,None].repeat(1,1,self.map.size(-1)) * self.map[None,:,:]).sum(dim=1)

        return finalP