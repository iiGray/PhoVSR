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


class RoPositionEncoding(nn.Module):
    def __init__(self,num_dims,dropout,max_len=1000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
        position=torch.arange(max_len,dtype=torch.float32)[:,None]
        inv_denominator=torch.exp(
            torch.arange(0,num_dims,2).float()\
            *-(4.*torch.log(torch.tensor(10.0))/num_dims)
        )

        self.sin=torch.sin(position*inv_denominator)
        self.cos=torch.cos(position*inv_denominator)
        
    def forward(self,x,current_steps=None):
        if self.sin.device!=x.device:
            self.sin=self.sin.to(x.device)
            self.cos=self.cos.to(x.device)
        x1,x2=x[..., 0::2], x[..., 1::2]
        if current_steps is None:
            sin,cos=self.sin[:x.size(2)][None,None,:,:],self.cos[:x.size(2)][None,None,:,:]
        else:
            sin,cos = self.sin[[current_steps]][None,None,:,:],self.cos[[current_steps]][None,None,:,:]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)
    
class AbsPositionEncoding(nn.Module):
    def __init__(self,num_dims,dropout,max_len=6000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
        position=torch.arange(max_len,dtype=torch.float32)[:,None]
        inv_denominator=torch.exp(
            torch.arange(0,num_dims,2).float()\
            *-(4.*torch.log(torch.tensor(10.0))/num_dims)
        )

        self.sin=torch.sin(position*inv_denominator)
        self.cos=torch.cos(position*inv_denominator)
        

    def forward(self,x,current_steps=None):
        if self.sin.device!=x.device:
            self.sin=self.sin.to(x.device)
            self.cos=self.cos.to(x.device)
        x1,x2=x[..., 0::2], x[..., 1::2]
        if current_steps is None:
            sin,cos=self.sin[:x.size(2)][None,None,:,:],self.cos[:x.size(2)][None,None,:,:]
        else:
            sin,cos=self.sin[[current_steps]][None,None,:,:],self.cos[[current_steps]][None,None,:,:]
        return torch.stack([x1 * sin + x2 * (-cos), x1 * (-cos) + x2 * (-sin)], dim=-1).flatten(-2)
    
class RoMultiHeadAttention(nn.Module):
    def __init__(self,num_dims,num_heads,dropout=0):
        super().__init__()
        assert num_dims%num_heads==0,"Input dims % num_heads != 0 !"
        self.num_dims=num_dims
        self.num_heads=num_heads
        self.per_head=num_dims//num_heads
        assert self.per_head%2==0,"RoPE needs the dim per head %2==0 !"

        self.dropout=nn.Dropout(dropout)

        for i in ["q","k","v","o"]:
            setattr(self,"W"+i,nn.Linear(num_dims,num_dims))
            # nn.init.xavier_uniform_(getattr(self,"W"+i).weight)

#         self.rel_enc=RoPositionEncoding(num_dims=self.per_head,dropout=dropout)
        self.abs_enc=AbsPositionEncoding(num_dims=self.per_head,dropout=dropout)

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
            neg_inf=torch.finfo(score.dtype).min
            score=score.masked_fill(mask,neg_inf)
            

        
        # if mask is not None:
        #     assert mask.shape[-2:]==score.shape[-2:],\
        #         f"Mask:{mask.shape}, Score:{score.shape}"
        score=torch.softmax(score,dim=-1)
        if mask is not None:
            score=score.masked_fill(mask,0.0)
        
        self.score=self.dropout(score)
        out=torch.matmul(self.score,V)
        out=self.Wo(out.transpose(1,2).reshape(V.size(0),-1,self.num_dims))
        
        return out
    
    def get_scores(self,Q,K):
        return torch.matmul(Q,K.transpose(-2,-1))


    def forward(self,q,k,v,mask=None,current_steps=None):
        '''
        valid_len shape: (bs,)
        '''
        b,_,_=q.shape
        
        Q,K,V=self.forward_qkv(q,k,v)
        
        PQ=self.abs_enc(Q,current_steps)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
        PK=self.abs_enc(K)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
#         wQ=self.Wm(PQ).view(b,q.size(1),-1,self.per_head).transpose(1,2)
#         wK=self.Wn(PK).view(b,k.size(1),-1,self.per_head).transpose(1,2)
#         RQ=(self.abs_enc(wQ) + PQ.view(b,q.size(1),-1,self.per_head).transpose(1,2))/2.
#         RK=(self.abs_enc(wK) + PK.view(b,k.size(1),-1,self.per_head).transpose(1,2))/2. #这个式子未测试:(xi Wq Fi + xi Wq Fi Wm Fi) (xj Wk Fj + xj Wk Fj Wn Fj) ^T
        
#         scores=self.get_scores(RQ,RK)
        
        scores=self.get_scores(PQ,PK)
        
        return self.forward_attention(scores,V,mask)
