from model.backend.roformer.attention import *
from model.backend.positionFFN import *

from template.e2e.utils import *
from template.e2e.utils import State

class RoformerLMLayer(nn.Module):
    def __init__(self,
                 idims,
                 hdims,
                 num_heads,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.enc_att=norm_type(
            idims,
            RoMultiHeadAttention(idims,num_heads,dropout),
            dropout
        )

        self.pos_ffn=PosWiseFFN(idims,hdims,dropout,norm_type)
    def forward(self, x ,mask, cache=None):
        if cache is None:
            x=self.enc_att(x, x, x, mask)
            x=self.pos_ffn(x)
        else:
            current_step=cache.size(1)
            q=x[:,[-1],:]
            q=self.enc_att(q, x, x, mask, current_step)            
            q=self.pos_ffn(q)
            x=torch.concat([cache,q], dim=1)

        return x



class RoformerLM(nn.Module,EvalModule):
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
        self.char_list_len=char_list_len


        self.tok_enc=nn.Sequential(
            nn.Embedding(char_list_len,idims//4),
            nn.Linear(idims//4,idims),
            nn.LayerNorm(idims),
            nn.ReLU()
            )

        self.encoder=nn.Sequential(*[
            RoformerLMLayer(idims,
                            hdims,
                            num_heads,
                            dropout,
                            norm_type
                            ) for _ in range(num_layers)
        ])

        self.fc=nn.Linear(idims,char_list_len)

    
    def forward(self,x,valid_len):
        x=self.tok_enc(x)
        
        mask=get_mask((x.size(1),x.size(1)),valid_len,zero_triu=True)

        for encoder_layer in self.encoder:
            x=encoder_layer(x,mask)

        out=self.fc(x)

        return out
    
    def initState(self, eState: State, *args, **kwargs) -> State:
        batch_beam_size=len(eState)
        self.caches=[torch.zeros((batch_beam_size,0,self.idims),
                                 dtype=eState.dtype,
                                 device=eState.device)\
                                    for _ in range(self.num_layers)]
    
        return eState
    
    def selectNext(self, bk_pref, bk_pred, *args, **kwargs):
        for i in range(len(self.caches)):
            self.caches[i]=State(self.caches[i]).setitem(bk_pref).feats

    def scoring(self, eState: State, dState: State):
        '''
        while evaluating:
        self.training = False
        '''
        x,x_valid_len=dState

        x=self.tok_enc(x)

        mask=None

        new_caches=[]

        for i,encoder_layer in enumerate(self.encoder):
            x=encoder_layer(x,mask,self.caches[i])
            new_caches.append(x)

        self.caches=new_caches


        return self.fc(x[:,-1,:])