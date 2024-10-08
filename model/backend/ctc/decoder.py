# -*- coding: utf-8 -*-
"""
    Author: iiGray
    Description: 
    
    We rewrite CTCdecoder to fit our frameword, see also in https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages,
    based on which , we add length mask and implement parallel batch beam search deocding, so that different length videos can be 
    decoded simultaneously.

    PhonemeCTCdecoder is our method ,based on CTCdecoder

    
    Note: Different sets of deocoding batches well lead to slightly different results, which is not because of the length of different
    masks, but cuda accuracy issues. 
"""

import torch
from torch import nn
if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())


from template.e2e.utils import State,EvalModule


class CTCdecoder(nn.Module,EvalModule):
    def __init__(self,
                 idims,
                 char_list_len,
                 blank=0):
        super().__init__()
        self.fc=nn.Linear(idims,char_list_len)
        self.blank=blank
    def forward(self,x,xl):
        return torch.log_softmax(self.fc(x),dim=-1)

    def initState(self, eState: State):
        x,xl=eState
        x=self.forward(x,xl)
        self.cState=State(x,xl)
        self.bState=State(x[:,:,[self.blank]].repeat(1,1,x.size(-1)),xl)
        
        mask=xl[:,None]>torch.arange(x.size(1),device=x.device)[None,:]
        self.mState=State(mask)

        return self.cState
    
    def preProcess(self, blank, sos, eos):
        '''
        init some ctc variables:
        
        lastP:   prefix P includes [pn,pb] , shapez 
        prefP:   prefix probs
        return cState: (batch_size=1, num_steps, vocab_size)
        '''
        #(batch_beam_size, num_steps, vocab_size)
        x=self.cState.feats

        self.logzero=-torch.finfo(torch.float16).min**2
#         self.logzero=-torch.inf
        self.blank=blank
        self.sos=sos
        self.eos=eos

        batch_size, num_steps, self.vocab_size = x.shape

        # (2, num_steps, batch_beam_size = batch_size), cur_beam = 1
        self.lastP=torch.full(
            (2, num_steps, batch_size),
            self.logzero,
            device=x.device
        )
        self.lastP[1]=torch.cumsum(x.transpose(0,1)[:,:,blank], dim=0)# (num_steps, batch_beam_size)

        self.prefP=0.
    
    def selectNext(self, bk_pref, bk_pred):
        '''
        select next cState after topk

        bk_pref:  (batch_beam_size, )  the  last   ids
        bk_pred:  (batch_beam_size, )  the current ids

        choose prefix from self.lastP and self.prefP
        
        self.lastP : (2, num_steps, batch_beam_size, ctc_beam_size) 
                  -> (2, num_steps, batch_beam_size)

        self.prefP : (num_steps, batch_beam_size, ctc_beam_size)
                  -> (num_steps, batch_beam_size) 
        '''

        self.cState.setitem(bk_pref)
        self.bState.setitem(bk_pref)
        self.mState.setitem(bk_pref)
        # (2, num_steps, batch_beam_size)
        bk_pred=self.vocabMap[bk_pref,bk_pred]
        
        # (num_steps, batch_beam_size, ctc_beam_size)
        self.prefP=self.prefP[bk_pref, bk_pred]\
            .unsqueeze(1).repeat(1,self.lastP.size(-1))
        
        self.lastP=self.lastP[:,:,bk_pref,bk_pred]



    def scoring(self,cState: State, dState: State, ctc_beam_idx):
        '''

        self.lastP  : (2, num_steps, batch_beam_size) the prefix probs of each num_steps
        self.prefP  : (batch_beam_size, 1)  the final prefix probs  

        cState      : initState   output (batch_beam_size, num_steps, vocab_size)
        bState      : blank prob in cState
        dState      : transformer output (batch_beam_size, cur_num_steps)
        ctc_beam_idx: (batch_beam_size, ctc_beam_size)
        return: ctc  score         (batch_beam_size, vocab_size)
        '''
        # already selected (batch_beam_size, cur_num_steps) 
        y,yl=dState
        #(batch_beam_size, num_steps, ctc_beam_size)
        x,xl=cState.select(ctc_beam_idx[:,None,:])
        xb,xbl=self.bState.select(ctc_beam_idx[:,None,:])
        #(batch_beam_size, num_steps, ctc_beam_size)
        batch_beam_size, num_steps, ctc_beam_size = x.shape
        X=torch.stack([x,xb],dim=0).transpose(1,2)
        # [xnb, xb] (2, num_steps, batch_beam_size, ctc_beam_size)
        
        # (num_steps, batch_beam_size)
        lastPsum=self.lastP.logsumexp(dim=0)
        
        #(batch_beam_size, 1)
        lastpred=y[:,[-1]]
        sameidx=(lastpred==ctc_beam_idx).nonzero(as_tuple=True)

        # (num_steps, batch_beam_size, ctc_beam_size)
        prefP=lastPsum[:,:,None].repeat(1,1,ctc_beam_size)
        prefP[:,sameidx[0],sameidx[1]]=self.lastP[1,:,sameidx[0]]

        # [pnb, pb] (2, num_steps, batch_beam_size, ctc_beam_size)
        P=torch.full(
            X.shape,
            self.logzero,
            dtype=X.dtype,
            device=X.device
        )
        # can only implement for recurrence
        start,end=min(y.size(1),x.size(1)-1), x.size(1)
        P[0, start] = X[0, start]
        for t in range(max(start,1),end):
            P[:,t]=torch.stack([P[0,t-1],prefP[t-1],
                                P[0,t-1],P[1,t-1]])\
                                    .view(2, 2,
                                          batch_beam_size,
                                          ctc_beam_size).logsumexp(1) + X[:, t]
                                          
        self.vocabMap=torch.full((batch_beam_size,self.vocab_size),-1,
                                 dtype=torch.long,device=P.device)
        
        self.vocabMap[torch.arange(batch_beam_size,device=P.device)[:,None],
                      ctc_beam_idx]=torch.arange(ctc_beam_size,device=P.device)



        # mask the non-valid part of prob vectors
        mask=xl[:,None]>torch.arange(x.size(1),device=x.device)[None,:]
        
        mask0=(xl-1)[:,None]>torch.arange(x.size(1),device=x.device)[None,:]

        curP=P.logsumexp(dim=0).transpose(0,1)
        curP.masked_fill_(~mask[:,:,None],self.logzero)
        curP=curP[:,start:end].logsumexp(dim=1)



        finalP=torch.full((curP.size(0),self.vocab_size),self.logzero,
                          dtype=curP.dtype,device=curP.device)
        
        finalP[torch.arange(ctc_beam_idx.size(0),
                            device=curP.device)[:,None],
               ctc_beam_idx]=curP-self.prefP 
        

        lastPsum=lastPsum.transpose(0,1)
        lastPsum.masked_fill_(~mask,0.)
        lastPsum.masked_fill_(mask0,0.)
        finalP[:,self.eos]=torch.sum(lastPsum,dim=1) 
        finalP[:,self.blank]=self.logzero

        '''
        self.lastP : (2, num_steps, batch_beam_size, ctc_beam_size)
        self.prefP : (num_steps, batch_beam_size, ctc_beam_size)
        
        they all need to be chosen in postProcess:

        self.lastP : (2, num_steps, batch_beam_size)
        self.prefP : (num_steps, batch_beam_size) 
        '''

        self.lastP=P

        self.prefP=curP
        
        return finalP
    

  
    
class PhonemeCTCdecoder(nn.Module,EvalModule):
    def __init__(self,
                 idims,
                 char_list_len,
                 g2p,
                 p2g,
                 blank=0):
        super().__init__()
        self.fc=nn.Linear(idims,char_list_len)
        self.blank=blank

        # g2p: shape: (len(graphemes list),len(phonemes list))  0-1
        # p2g: phoenem idx to grapheme idx
        self.g2p=g2p
        self.p2g=p2g
        
        self.mask_inf=self.p2w*1.
        self.mask_inf[self.mask_inf==0]= -torch.finfo(torch.float16).min**2
        self.mask_inf[self.mask_inf==1]= 0

    def forward(self,x,xl):
        return torch.log_softmax(self.fc(x),dim=-1)

    def initState(self, eState: State):
        x,xl=eState
        x=self.forward(x,xl)
        self.cState=State(x,xl)
        self.bState=State(x[:,:,[self.blank]].repeat(1,1,x.size(-1)),xl)
        
        self.w2p=self.w2p.to(eState.device)
        self.p2w=self.p2w.to(eState.device)
        self.mask_inf=self.mask_inf.to(eState.device)
        self.ctc_arange=torch.arange(self.p2w.size(0),device=eState.device)

        return self.cState
    
    def preProcess(self, blank, sos, eos):
        '''
        init some ctc variables:
        
        lastP:   prefix P includes [pn,pb] , shapez 
        prefP:   prefix probs
        eosP :   prefix probs of eos
        return cState: (batch_size=1, num_steps, vocab_size)
        '''
        #(batch_beam_size, num_steps, vocab_size)
        x=self.cState.feats

        self.logzero=-torch.finfo(torch.float16).min**2
#         self.logzero=-torch.inf
        self.blank=blank
        self.sos=sos
        self.eos=eos
        
        batch_size, num_steps, self.vocab_size = x.shape

        # (2, num_steps, batch_beam_size = batch_size), cur_beam = 1
        self.lastP=torch.full(
            (2, num_steps, batch_size),
            self.logzero,
            device=x.device
        )
        self.lastP[1]=torch.cumsum(x.transpose(0,1)[:,:,blank], dim=0)# (num_steps, batch_beam_size)

        self.prefP=0.

        self.last_pred=torch.ones((x.size(0)),
                                  dtype=torch.long,
                                  device=x.device)*self.sos

        # (2, num_steps, batch_beam_size = batch_size), cur_beam = 1
        self.eosP=torch.full(
            (2, num_steps, batch_size),
            self.logzero,
            device=x.device
        )

    def selectNext(self, bk_pref, bk_pred):
        '''
        select next cState after topk

        bk_pref:  (batch_beam_size, )  the  last   ids
        bk_pred:  (batch_beam_size, )  the current phoneme ids

        choose prefix from self.lastP and self.prefP

        self.matP  : (batch_beam_size, phoneme_vocab_size, vocab_size)
        
        self.lastP : (2, num_steps, batch_beam_size, ctc_beam_size) 
                  -> (2, num_steps, batch_beam_size)

        self.prefP : (num_steps, batch_beam_size, ctc_beam_size)
                  -> (num_steps, batch_beam_size) 
        '''
        
        self.cState.setitem(bk_pref)
        self.bState.setitem(bk_pref)
        # (2, num_steps, batch_beam_size)

        # (batch_beam_size, phoneme_vocab_size, ) -> (batch_beam_size, )
        '''
        #origin:
        bk_pred=(self.matP+self.mask_inf[None,:,:])[bk_pref,:,bk_pred].argmax(dim=1)
        
        bk_pred=(self.matP+self.mask_inf[None,:,:]).argmax(dim=1)[bk_pref,bk_pred]
        '''
        #new
        bk_pred=self.max_arg[bk_pref,bk_pred]
        
        self.last_pred=bk_pred
        
        # (batch_beam_size, ctc_arange_size)
        self.prefP=self.prefP[bk_pref, bk_pred]\
            .unsqueeze(1).repeat(1,self.lastP.size(-1))
        
        # (2, num_steps, batch_beam_size)
        self.lastP=self.lastP[:,:,bk_pref,bk_pred]



    def scoring(self,cState: State, dState: State, ctc_beam_idx):
        '''

        self.lastP  : (2, num_steps, batch_beam_size) the prefix probs of each num_steps
        self.prefP  : (batch_beam_size, 1)  the final prefix probs  

        cState      : initState   output (batch_beam_size, num_steps, vocab_size)
        bState      : blank prob in cState
        dState      : transformer output (batch_beam_size, cur_num_steps)
        ctc_beam_idx: (batch_beam_size, ctc_beam_size) ,but not use

        ctc_arange_idx: all ctc idx
        vocabMap    :  vocab_ids(phoneme) to beam_ids
        return: ctc  score         (batch_beam_size, vocab_size)
        '''
        # already selected (batch_beam_size, cur_num_steps) 
        y,yl=dState
        #(phoneme_vocab_size, ) = (ctc_arange_size, )
        ctc_arange_idx=self.ctc_arange
        
        x,xl=cState
        xb,xbl=self.bState
        #(batch_beam_size, num_steps, ctc_arange_size)
        batch_beam_size, num_steps, ctc_arange_size = x.shape
        X=torch.stack([x,xb],dim=0).transpose(1,2)
        # [xnb, xb] (2, num_steps, batch_beam_size, ctc_arange_size)
        
        # (num_steps, batch_beam_size)
        lastPsum=self.lastP.logsumexp(dim=0)
        
        #(batch_beam_size, 1)
        lastpred=self.last_pred[:,None]

        sameidx=(lastpred==ctc_arange_idx).nonzero(as_tuple=True)

        # (num_steps, batch_beam_size, ctc_arange_size)
        prefP=lastPsum[:,:,None].repeat(1,1,ctc_arange_size)
        prefP[:,sameidx[0],sameidx[1]]=self.lastP[1,:,sameidx[0]]

        # [pnb, pb] (2, num_steps, batch_beam_size, ctc_arange_size)
        P=torch.full(
            X.shape,
            self.logzero,
            dtype=X.dtype,
            device=X.device
        )
        # can only implement for recurrence
        start,end=min(y.size(1)-1,x.size(1)-1), x.size(1)
        P[0, start] = X[0, start]
        for t in range(max(start,1),end):
            P[:,t]=torch.stack([P[0,t-1],prefP[t-1],
                                P[0,t-1],P[1,t-1]])\
                                    .view(2, 2,
                                          batch_beam_size,
                                          ctc_arange_size).logsumexp(1) + X[:, t]


        # mask the non-valid part of prob vectors
        mask=xl[:,None]>torch.arange(x.size(1),device=x.device)[None,:]
        
        mask0=(xl-1)[:,None]>torch.arange(x.size(1),device=x.device)[None,:]

        curP=P.logsumexp(dim=0).transpose(0,1)
        curP.masked_fill_(~mask[:,:,None],self.logzero)
        # (batch_beam_size, ctc_arange_size)
        curP=curP[:,start:end].logsumexp(dim=1)


        # (batch_beam_size, phoneme_vocab_size, vocab_size)

        self.matP=(curP-self.prefP)[:,:,None].repeat(1,1,self.p2w.size(-1)) * self.p2w[None,:,:] 
        
#         self.matP[:,[0,1,2,3],:]=-torch.inf

        # (batch_beam_size, vocab_size)

        finalP,self.max_arg=(self.matP+self.mask_inf[None,:,:]).max(dim=1)
#         finalP=self.matP.sum(dim=1)
        
        # add eos to score , and return a 3363 len vector,


        lastPsum=lastPsum.transpose(0,1)
        lastPsum.masked_fill_(~mask,0.)
        lastPsum.masked_fill_(mask0,0.)
        finalP[:,self.eos]=torch.sum(lastPsum,dim=1)
        # finalP[:,self.eos]=lastPsum[num_steps-1,:]
        finalP[:,[self.blank,1]]=self.logzero

        '''
        self.lastP : (2, num_steps, batch_beam_size, ctc_arange_size)
        self.prefP : (num_steps, batch_beam_size, ctc_arange_size)
        
        they all need to be chosen in postProcess:

        self.lastP : (2, num_steps, batch_beam_size)
        self.prefP : (num_steps, batch_beam_size) 
        '''

        self.lastP=P

        self.prefP=curP
        
        return finalP#/end
    
if __name__=="__main__":
    a=torch.randn(2,4,2,10)
    b=torch.tensor([[2,3],
                    [1,4]])
    print(a[:,:,torch.arange(b.size(0))[:,None],b].shape)