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
    
#     def get_scores(self,Q,K,WR):
#         AC=torch.matmul(Q+self.u,K.transpose(-2,-1))
#         BD=torch.matmul(Q+self.v,WR.transpose(-2,-1))
#         BD=self.rel_shift(BD)

#         BD=BD[:,:,:,BD.size(-1)//2+1-AC.size(-1):BD.size(-1)//2+1]
            
#         scores=(AC+BD)/self.per_head**0.5
#         return scores
    
#     def forward(self,q,k,v,R, mask,cache=None,cache_mask=None):
     
# #         if cache is not None:
# #             k=v=torch.concat([cache,k],dim=1)
# #             mask=torch.concat([mask,cache_mask],dim=-1) #(b,h,1,t)

#         Q,K,V=self.forward_qkv(q,k,v)

#         WR=self.forward_r(R)

#         scores=self.get_scores(Q,K,WR)

#         return self.forward_attention(scores,V,mask)


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

#     def get_scores(self,Q,K,WR):
#         AC=torch.matmul(Q+self.u,K.transpose(-2,-1))
#         BD=torch.matmul(Q+self.v,WR.transpose(-2,-1))
#         BD=self.rel_shift(BD)

#         BD=BD[:,:,:,:BD.size(-1)//2+1]
        
#         scores=(AC+BD)/self.per_head**0.5
#         return scores
#     def forward(self, q, k, v, R, mask=None, cache=None, cache_mask=None):
#         return super().forward(q, k, v, R, mask, cache, cache_mask)

            

if __name__=="__main__":

    f=get_mask((5,6),torch.tensor([3,2,5,4]),zero_triu=True)
    print(f)
    # print(torch.randn(6,6).masked_fill(f,torch.finfo(torch.float32).min))

# '''official'''
# class MultiHeadedAttention(nn.Module):
#     """Multi-Head Attention layer.
#     Args:
#         n_head (int): The number of heads.
#         n_feat (int): The number of features.
#         dropout_rate (float): Dropout rate.
#     """

#     def __init__(self, n_head, n_feat, dropout_rate):
#         """Construct an MultiHeadedAttention object."""
#         super(MultiHeadedAttention, self).__init__()
#         assert n_feat % n_head == 0
#         # We assume d_v always equals d_k
#         self.d_k = n_feat // n_head
#         self.h = n_head
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         self.linear_k = nn.Linear(n_feat, n_feat)
#         self.linear_v = nn.Linear(n_feat, n_feat)
#         self.linear_o = nn.Linear(n_feat, n_feat)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward_qkv(self, query, key, value):
#         """Transform query, key and value.
#         Args:
#             query (torch.Tensor): Query tensor (#batch, time1, size).
#             key (torch.Tensor): Key tensor (#batch, time2, size).
#             value (torch.Tensor): Value tensor (#batch, time2, size).
#         Returns:
#             torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
#             torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
#             torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
#         """
#         n_batch = query.size(0)
#         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
#         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
#         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
#         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
#         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
#         v = v.transpose(1, 2)  # (batch, head, time2, d_k)

#         return q, k, v

#     def forward_attention(self, value, scores, mask, rtn_attn=False):
#         """Compute attention context vector.
#         Args:
#             value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
#             scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
#             mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
#             rtn_attn (boolean): Flag of return attention score
#         Returns:
#             torch.Tensor: Transformed value (#batch, time1, d_model)
#                 weighted by the attention score (#batch, time1, time2).
#         """
#         n_batch = value.size(0)
#         if mask is not None:
#             import numpy
#             mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
#             min_value = float(
#                 numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
#             )
#             scores = scores.masked_fill(mask, min_value)
#             self.attn = torch.softmax(scores, dim=-1).masked_fill(
#                 mask, 0.0
#             )  # (batch, head, time1, time2)
#         else:
#             self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

#         p_attn = self.dropout(self.attn)
#         x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
#         x = (
#             x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
#         )  # (batch, time1, d_model)
#         if rtn_attn:
#             return self.linear_o(x), self.attn
#         return self.linear_o(x)  # (batch, time1, d_model)

#     def forward(self, query, key, value, mask, rtn_attn=False):
#         """Compute scaled dot product attention.
#         Args:
#             query (torch.Tensor): Query tensor (#batch, time1, size).
#             key (torch.Tensor): Key tensor (#batch, time2, size).
#             value (torch.Tensor): Value tensor (#batch, time2, size).
#             mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
#                 (#batch, time1, time2).
#             rtn_attn (boolean): Flag of return attention score
#         Returns:
#             torch.Tensor: Output tensor (#batch, time1, d_model).
#         """
#         q, k, v = self.forward_qkv(query, key, value)
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         return self.forward_attention(v, scores, mask, rtn_attn)
# def valid_attention():
#     m1=MultiHeadAttention(40,4)
#     m2=MultiHeadedAttention(4,40,0)
#     for k,v in m1.state_dict().items():
#         setattr(m1,k[:2],getattr(m2,"linear_"+k[1]))
#     q=torch.randn(3,50,40)
#     valid_len=torch.tensor([6,4,8])
#     a=m1(q,q,q,valid_len)
#     b=m2(q,q,q,m1.mask[:,0,:,:])
#     print((a==b).sum(),a.shape)
# class RelPositionalEncoding(torch.nn.Module):
#     """Relative positional encoding module (new implementation).
#     Details can be found in https://github.com/espnet/espnet/pull/2816.
#     See : Appendix B in https://arxiv.org/abs/1901.02860
#     Args:
#         d_model (int): Embedding dimension.
#         dropout_rate (float): Dropout rate.
#         max_len (int): Maximum input length.
#     """

#     def __init__(self, d_model, dropout_rate, max_len=5000):
#         """Construct an PositionalEncoding object."""
#         super(RelPositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.xscale = math.sqrt(self.d_model)
#         self.dropout = torch.nn.Dropout(p=dropout_rate)
#         self.pe = None
#         self.extend_pe(torch.tensor(0.0).expand(1, max_len))

#     def extend_pe(self, x):
#         """Reset the positional encodings."""
#         if self.pe is not None:
#             # self.pe contains both positive and negative parts
#             # the length of self.pe is 2 * input_len - 1
#             if self.pe.size(1) >= x.size(1) * 2 - 1:
#                 if self.pe.dtype != x.dtype or self.pe.device != x.device:
#                     self.pe = self.pe.to(dtype=x.dtype, device=x.device)
#                 return
#         # Suppose `i` means to the position of query vecotr and `j` means the
#         # position of key vector. We use position relative positions when keys
#         # are to the left (i>j) and negative relative positions otherwise (i<j).
#         pe_positive = torch.zeros(x.size(1), self.d_model)
#         pe_negative = torch.zeros(x.size(1), self.d_model)
#         position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, self.d_model, 2, dtype=torch.float32)
#             * -(math.log(10000.0) / self.d_model)
#         )
#         pe_positive[:, 0::2] = torch.sin(position * div_term)
#         pe_positive[:, 1::2] = torch.cos(position * div_term)
#         pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
#         pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

#         # Reserve the order of positive indices and concat both positive and
#         # negative indices. This is used to support the shifting trick
#         # as in https://arxiv.org/abs/1901.02860
#         pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
#         pe_negative = pe_negative[1:].unsqueeze(0)
#         pe = torch.cat([pe_positive, pe_negative], dim=1)
#         self.pe = pe.to(device=x.device, dtype=x.dtype)

#     def forward(self, x: torch.Tensor):
#         """Add positional encoding.
#         Args:
#             x (torch.Tensor): Input tensor (batch, time, `*`).
#         Returns:
#             torch.Tensor: Encoded tensor (batch, time, `*`).
#         """
#         self.extend_pe(x)
#         x = x * self.xscale
#         pos_emb = self.pe[
#             :,
#             self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
#         ]
#         return self.dropout(x), self.dropout(pos_emb)
# def valid_posEncoding():
#     relpos1=RelPositionEncoding(40)
#     relpos2=RelPositionalEncoding(40,dropout_rate=0)
#     x=torch.randn(5,10,40)
#     a=relpos1(x)
#     b=relpos2(x)
#     # print(a[1]==b[1])
#     print((a[0]==b[0]).sum(),a[0].shape)
#     print((a[1]==b[1]).sum(),a[1].shape)
# class RelPositionMultiHeadedAttention(MultiHeadedAttention):
#     """Multi-Head Attention layer with relative position encoding (new implementation).
#     Details can be found in https://github.com/espnet/espnet/pull/2816.
#     Paper: https://arxiv.org/abs/1901.02860
#     Args:
#         n_head (int): The number of heads.
#         n_feat (int): The number of features.
#         dropout_rate (float): Dropout rate.
#         zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
#     """

#     def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
#         """Construct an RelPositionMultiHeadedAttention object."""
#         super().__init__(n_head, n_feat, dropout_rate)
#         self.zero_triu = zero_triu
#         # linear transformation for positional encoding
#         self.linear_r = nn.Linear(n_feat, n_feat, bias=False)
#         # these two learnable bias are used in matrix c and matrix d
#         # as described in https://arxiv.org/abs/1901.02860 Section 3.3
#         self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
#         self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
#         torch.nn.init.xavier_uniform_(self.pos_bias_u)
#         torch.nn.init.xavier_uniform_(self.pos_bias_v)

#     def rel_shift(self, x):
#         """Compute relative positional encoding.
#         Args:
#             x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
#             time1 means the length of query vector.
#         Returns:
#             torch.Tensor: Output tensor.
#         """
#         zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
#         x_padded = torch.cat([zero_pad, x], dim=-1)

#         x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
#         x = x_padded[:, :, 1:].view_as(x)[
#             :, :, :, : x.size(-1) // 2 + 1
#         ]  # only keep the positions from 0 to time2

#         if self.zero_triu:
#             ones = torch.ones((x.size(2), x.size(3)), device=x.device)
#             x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        
#         return x

#     def forward(self, query, key, value, pos_emb, mask):
#         """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
#         Args:
#             query (torch.Tensor): Query tensor (#batch, time1, size).
#             key (torch.Tensor): Key tensor (#batch, time2, size).
#             value (torch.Tensor): Value tensor (#batch, time2, size).
#             pos_emb (torch.Tensor): Positional embedding tensor
#                 (#batch, 2*time1-1, size).
#             mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
#                 (#batch, time1, time2).
#         Returns:
#             torch.Tensor: Output tensor (#batch, time1, d_model).
#         """
#         q, k, v = self.forward_qkv(query, key, value)
#         q = q.transpose(1, 2)  # (batch, time1, head, d_k)

#         n_batch_pos = pos_emb.size(0)
#         p = self.linear_r(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
#         p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

#         # (batch, head, time1, d_k)
#         q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
#         # (batch, head, time1, d_k)
#         q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

#         # compute attention score
#         # first compute matrix a and matrix c
#         # as described in https://arxiv.org/abs/1901.02860 Section 3.3
#         # (batch, head, time1, time2)
#         matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

#         # compute matrix b and matrix d
#         # (batch, head, time1, 2*time1-1)
#         matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
#         matrix_bd = self.rel_shift(matrix_bd)

#         scores = (matrix_ac + matrix_bd) / math.sqrt(
#             self.d_k
#         )  # (batch, head, time1, time2)

#         return self.forward_attention(v, scores, mask)
# def valid_relattention():
#     # r1=DecoderRelPosMultiHeadAttn(40,4)
#     r1=EncoderRelPosMultiHeadAttn(40,4,use_cache=False)
#     # r1=RelPosMultiHeadAttention(40,4,use_cache=False)
#     r2=RelPositionMultiHeadedAttention(4,40,0)

#     q=torch.randn(2,4,40)
#     r=torch.randn(2,7,40)
#     for i in ["q","k","v","o","r",]:
#         setattr(r1,"W"+i,getattr(r2,"linear_"+i))
    
#     r1.u.data=r2.pos_bias_u.data.view_as(r1.u.data)
#     r1.v.data=r2.pos_bias_v.data.view_as(r1.v.data)
#     # print(r1.u.data.shae)
#     a=r1(q,q,q,r)
#     b=r2(q,q,q,r,None)
#     print((a==b).sum(),a.shape)
    
# def valid_cahceAttn():
#     r1=RelPosMultiHeadAttention(40,4,use_cache=True)
#     q=torch.randn(2,4,40)
#     r=torch.randn(2,7,40)
#     # print(r1.use_cache)
#     print(r1(q,q,q,r,cache=torch.randn(2,3,40)).shape)

if __name__=="__main__":
    # de=EncoderRelPosMultiHeadAttn(40,4)

    # d=DecoderRelPosMultiHeadAttn(40,4,mode="src")
    # q=torch.randn(2,4,40)
    # r=torch.randn(2,7,40)
    # valid_len=torch.tensor([3,4]).long()
    # print(d(q,q,q,r,valid_len).shape)
    # print(de(q,q,q,r,valid_len).shape)
    # valid_cahceAttn()
    # valid_relattention()
    # valid_attention()
    # valid_posEncoding()
    pass
