import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from typing import Literal

class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    def get_mask(self,t,valid_len):
        if valid_len is None:
            return None
        mask=(torch.arange(t).to(valid_len.device)[None,:]<valid_len[:,None]) # (b, t)
        return mask
    
    def forward(self,x,label,valid_len=None):
        '''
        x     shape: (b,t,f)
        label shape:(b,t, )
        '''
        reduction=self.reduction
        self.reduction="none"
        loss=super().forward(x.permute(0,2,1),label)
        if valid_len is not None:
            mask=self.get_mask(x.size(1),valid_len)
            loss=loss*mask
        else:
            valid_len=torch.tensor(x.size(0)*x.size(1),
                                   device=x.device)
        
        self.reduction=reduction
        
        if reduction=="sum":
            return loss.sum()
        if reduction=="mean":
            return loss.sum()/valid_len.sum()
        
        return loss

class MaskedLabelSmoothingLoss(nn.Module):
    def __init__(self,
                 ignore_idx,
                 smoothing=0.0,
                 reduction:Literal["mean","sum","none","batchmean"]="mean"
                 ):
        super().__init__()
        self.criterion=nn.KLDivLoss(reduction="none")
        self.ignore_idx=ignore_idx
        self.confidence=1.0 - smoothing
        self.smoothing=smoothing
        self.true_dist=None
        self.reduction=reduction
    def get_mask(self,t,valid_len):
        if valid_len is None:
            return None
        mask=(torch.arange(t).to(valid_len.device)[None,:]<valid_len[:,None]) # (b, t)
        return mask

    def acc(self,x,label,valid_len):
        pass        

    def forward(self,x,label,valid_len=None):
        '''
        x      : (b, t, f)
        label  : (b, t, )
        '''
        batch_size,num_steps,vocab_size=x.shape
        
        with torch.no_grad():
            true_dist=x.clone()
            true_dist.fill_(self.smoothing/(vocab_size-2)) #exclude self and ignore_idx
            true_dist[:, :, self.ignore_idx]=0
            true_dist.scatter_(-1, label.unsqueeze(-1),self.confidence)
            
        kl=self.criterion(x.log_softmax(dim=-1),
                          true_dist).sum(-1) #(b, t, f)  -> (b, t,)
        if valid_len is not None:
            mask=self.get_mask(num_steps,valid_len)
            kl=kl*mask
        else:
            valid_len=torch.tensor(x.size(0)*x.size(1),
                                   device=x.device)
        
        if self.reduction=="sum":
            return kl.sum()
        if self.reduction=="batchmean":
            return kl.sum()/x.size(0)
        
        if self.reduction=="mean":
            return kl.sum()/valid_len.sum()
        
        return kl   

    
class CTCLoss(nn.Module):
    def __init__(self,blank=0,reduction:Literal["none","sum","mean"]="mean"):
        super().__init__()
        self.blank=blank
        self.reduction=reduction
        
    
    def lower_bound(self,target,target_length,BT,T,device):
        same= target==torch.concat([torch.tensor([-1],device=device),target[:-1]])

        descend_len=target_length+same.sum().item()
        descend_seg=torch.zeros((descend_len,),dtype=target.dtype,device=device)
        even_ids=torch.nonzero(same).flatten()
        even_arange=torch.arange(even_ids.size(0),device=device)
        even_val=even_ids*2
        even_ids=even_ids+even_arange
        descend_seg[even_ids]=even_val
        descend_seg[descend_seg==0]=torch.arange(target_length,device=device)*2+1
        if T<descend_seg.size(0):
            return torch.tensor([torch.nan],device=target.device)
        lower_bound=torch.concat([descend_seg,torch.ones((T-descend_seg.size(0)),
                                                         dtype=descend_seg.dtype,
                                                         device=device)*(BT-1)])
        return lower_bound
        
    def forward_one_sample(self,mat,target,blanked_target):
        '''
        mat           : (2*target_length+1, input_length)
        target        : (target_length, )
        blanked_target: (2*target_length+1, )
        
        return : torch.Size([])
        '''
        BT,T=mat.shape # blank T, orign T

        #compute lower bound
        lower_bound=self.lower_bound(target,target.size(0),BT,T,mat.device)
        #compute upper bound
        upper_bound=BT-1-self.lower_bound(target.flip(dims=[0]),
                                          target.size(0),
                                          BT,T,mat.device).flip(dims=[0])
        
        if lower_bound[0].isnan().item() or upper_bound[0].isnan().item() \
        or lower_bound[-1].item()<upper_bound[-1].item():
            return torch.tensor(torch.inf,dtype=mat.dtype,device=mat.device)
        
        preup,prelo=upper_bound[0].item(),lower_bound[0].item()
        logzero=torch.tensor(-torch.inf,dtype=mat.dtype,device=mat.device)
        
        rmat=[[None]*mat.size(1) for _ in range(mat.size(0))]
        rmat[0][0]=mat[0,0]
        rmat[1][0]=mat[1,0]
        for i in range(1,T):
            up,lo=upper_bound[i].item(),lower_bound[i].item()
            for j in range(up,lo+1):
                pre_prob=torch.logaddexp((rmat[j][i-1] if preup<=j<=prelo else logzero),
                                         (rmat[j-1][i-1] if preup<=j-1<=prelo else logzero))
                if j&1 and preup<=j-2<=prelo and (blanked_target[j-2]!=blanked_target[j]).item(): # pos j is not blank
                    pre_prob=torch.logaddexp(pre_prob,rmat[j-2][i-1])
                rmat[j][i]=mat[j,i]+pre_prob
            preup,prelo=up,lo

        ret=logzero
        for j in range(up,lo+1):
            ret=torch.logaddexp(ret,rmat[j][T-1])
        return ret
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        '''
        log_probs     : (ipt_num_steps, batch_size, vocab_size)
        targets       : (batch_size, tgt_num_steps)
        input_lengths : (batch_size, )
        target_lengths: (batch_size, )
        '''
        B=log_probs.size(1)
        blanks=torch.ones_like(targets,dtype=targets.dtype,device=targets.device)*self.blank
        blanked_tgts=torch.stack([targets,blanks],dim=2).flatten(-2) # (B, tgt_T, 2)
        blanked_tgts=torch.concat([blanks[:,[0]],blanked_tgts],dim=1) #(B, tgt_T*2+1)
        blanked_prob=log_probs.permute(1,2,0)[torch.arange(B,device=blanks.device)[:,None],blanked_tgts] # (B, tgt_T*2+1, T)
        blanked_prob=blanked_prob.permute(1,0,2) # (tgt2_T*2+1, B, T)
        unpadded_prob=rnn_utils.unpad_sequence(blanked_prob,(target_lengths*2+1).cpu())
        
        ret=torch.stack([
            self.forward_one_sample(mat[:,:input_lengths[i]],
                                    targets[i][:target_lengths[i]],
                                    blanked_tgts[i])\
            for i,mat in enumerate(unpadded_prob)
        ])
        
    
        if self.reduction=="sum":
            return ret.sum()
        if self.reduction=="mean":
            return (ret/target_lengths/B).sum()
        return -ret    
    

    