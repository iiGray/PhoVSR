import torch
from torch import nn
from einops import rearrange,repeat
import math
from typing import Literal

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())


from model.backend.normalize import *

'''self'''

def get_mask(t,valid_len,zero_triu):
    '''
    t        : int or tuple  (q len,k len)
    valid_len: (b, ) (k len)
    '''
    qt,kt=(1,t) if isinstance(t,int) else t

    if valid_len is None:
        return None
    if valid_len.ndim>2:# means valid_len is actually a mask not len 
        return valid_len
    mask=(torch.arange(kt).to(valid_len.device)[None,:]<valid_len[:,None]) # (b, kt)
    mask=mask[:,None,:].repeat(1,qt,1) # (b, qt, kt)
    if zero_triu: # decoder self attention
        mask=torch.tril(mask)
    mask=mask[:,None,:,:] # (b, 1(h), qt, kt)
    return ~mask

class MultiHeadAttention(nn.Module):
    def __init__(self,num_dims,num_heads,dropout=0):
        super().__init__()
        assert num_dims%num_heads==0,"Input dims % num_heads != 0 !"
        self.num_dims=num_dims
        self.num_heads=num_heads
        self.per_head=num_dims//num_heads
        for i in ["q","k","v","o"]:
            setattr(self,"W"+i,nn.Linear(num_dims,num_dims))
            # nn.init.xavier_uniform_(getattr(self,"W"+i).weight)

        self.dropout=nn.Dropout(dropout)

    def forward_qkv(self,q,k,v):
        b,_,_=q.shape
        Q=self.Wq(q).view(b,q.size(1),-1,self.per_head).transpose(1,2)
        K=self.Wk(k).view(b,k.size(1),-1,self.per_head).transpose(1,2)
        V=self.Wv(v).view(b,v.size(1),-1,self.per_head).transpose(1,2)
        return Q,K,V
    
    def forward_attention(self,QK,V,mask=None):
        if type(QK) is tuple:
            Q,K=QK
            score=torch.matmul(Q,K.transpose(-2,-1))/self.per_head**0.5
            
        else:score=QK

        
        if mask is not None:
#             neg_inf=torch.finfo(score.dtype).min
            neg_inf=-torch.inf
            score=score.masked_fill(mask,neg_inf)
            
        score=torch.softmax(score,dim=-1)
        if mask is not None:
            score=score.masked_fill(mask,0.0)
        
        self.score=self.dropout(score)
        out=torch.matmul(self.score,V)
        out=self.Wo(out.transpose(1,2).reshape(V.size(0),-1,self.num_dims))
        
        return out

    def forward(self,q,k,v,mask=None):
        '''
        valid_len shape: (bs,)
        '''
        Q,K,V=self.forward_qkv(q,k,v)
        return self.forward_attention((Q,K),V,mask)
    
    
class PositionEncoding(nn.Module):
    def __init__(self,num_dims,dropout,max_len=8000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_dims))
        
        position=torch.arange(max_len,dtype=torch.float32)[:,None]
        inv_denominator=torch.exp(
            torch.arange(0,num_dims,2).float()\
            *-(4.*torch.log(torch.tensor(10.0))/num_dims)
        )
        self.P[:,:,0::2]=torch.sin(position*inv_denominator)
        self.P[:,:,1::2]=torch.cos(position*inv_denominator)
        
    def forward(self,x):
        x=x+self.P[:,:x.size(1),:].to(dtype=x.dtype,device=x.device)
        return self.dropout(x)

class RelPositionEncoding(nn.Module):
    def __init__(self,num_dims,dropout=0,max_len=8000):
        super().__init__()
        self.num_dims=num_dims
        self.dropout=nn.Dropout(dropout)

        max_len//=2
        positive=torch.zeros(max_len,num_dims)
        negative=torch.zeros(max_len,num_dims)
        position=torch.arange(0,max_len).float()[:,None]
        inv_denominator=torch.exp(
            torch.arange(0,num_dims,2).float()\
            *-(4.*torch.log(torch.tensor(10.0))/num_dims)
        )

        positive[:,0::2]=torch.sin(position*inv_denominator)
        positive[:,1::2]=torch.cos(position*inv_denominator)
        negative[:,0::2]=torch.sin(-position*inv_denominator)
        negative[:,1::2]=torch.cos(-position*inv_denominator)
        positive=positive.flip([0])[None,:,:]
        negative=negative[1:][None,:,:]

        self.P=torch.cat([positive,negative],dim=1)#(1,max_len,num_dims)
    

    def forward(self,x):
        x=x*self.num_dims**0.5
        assert self.P.size(1)>2*x.size(1)+1, "PosEncoding dim Not Enough!"
        P=self.P[:,self.P.size(1)//2-x.size(1)+1:self.P.size(1)//2+x.size(1)]
        return self.dropout(x),self.dropout(P.to(dtype=x.dtype,device=x.device))

class RelPosMultiHeadAttention(MultiHeadAttention):
    def __init__(self,num_dims,num_heads,dropout=0):
        super().__init__(num_dims,num_heads,dropout)
        
        self.Wr=nn.Linear(num_dims,num_dims,bias=False)
        self.u=nn.Parameter(torch.Tensor(1,self.num_heads,1,self.per_head))
        self.v=nn.Parameter(torch.Tensor(1,self.num_heads,1,self.per_head))        
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

    def forward_r(self,r):
        b,t,f=r.shape
        R=self.Wr(r).view(b,t,-1,self.per_head).transpose(1,2)
        return R

    def rel_shift(self,M):
        '''
        M shape: (b,h,t1,t2) , where t1 = tq, t2= 2 * tk - 1
        '''
        b,h,t1,t2=M.shape
        zeros=torch.zeros((b,h,t1,1),dtype=M.dtype,device=M.device)
        M=torch.concat([zeros,M],dim=-1)
        M=M.reshape(b,h,t2+1,t1)[:,:,1:].view(b,h,t1,t2)
        M=torch.tril(M,diagonal=t2-t1)
        return M
    def get_scores(self,Q,K,WR,current_step=None):
        AC=torch.matmul(Q+self.u,K.transpose(-2,-1))
        if current_step is None:
            BD=torch.matmul(Q+self.v,WR.transpose(-2,-1))
            BD=self.rel_shift(BD)
            BD=BD[:,:,:,BD.size(-1)//2+1-AC.size(-1):BD.size(-1)//2+1]
        else:
            cWR=WR[:,:,K.size(-2)-current_step:2*K.size(-2)-current_step]
            BD=torch.matmul(Q+self.v,cWR.transpose(-2,-1))

        
        scores=(AC+BD)/self.per_head**0.5
        
        return scores
    
    def forward(self,q,k,v,R, mask,current_step=None):

        Q,K,V=self.forward_qkv(q,k,v)

        WR=self.forward_r(R)

        scores=self.get_scores(Q,K,WR,current_step)

        return self.forward_attention(scores,V,mask)


class EncoderRelPosMultiHeadAttn(RelPosMultiHeadAttention):
    def __init__(self,
                 num_dims,
                 num_heads,
                 dropout=0,
                 ):
        super().__init__(num_dims,num_heads,dropout)
    
    def get_scores(self,Q,K,WR,current_step):
        AC=torch.matmul(Q+self.u,K.transpose(-2,-1))
        if current_step is None:
            BD=torch.matmul(Q+self.v,WR.transpose(-2,-1))
            BD=self.rel_shift(BD)
            BD=BD[:,:,:,:BD.size(-1)//2+1]
        else:
            cWR=WR[:,:,K.size(-2)-current_step:2*K.size(-2)-current_step]
            BD=torch.matmul(Q+self.v,cWR.transpose(-2,-1))

        
        scores=(AC+BD)/self.per_head**0.5
        
        self.attention_weights=scores
        
        return scores



if __name__=="__main__":

    f=get_mask((5,6),torch.tensor([3,2,5,4]),zero_triu=True)
    print(f)
    # print(torch.randn(6,6).masked_fill(f,torch.finfo(torch.float32).min))

