import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from einops import rearrange
from typing import Dict,NamedTuple,Tuple,Optional,Union,List
from collections import defaultdict

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())


from template.e2e.search.utils import Prefix,EndedPrefixes
from template.e2e.search.utils import *
from template.e2e.utils import State,Scorer,EvalModule


''' in evaluator'''

class DecodeSearch:
    '''
    By inheriting it, you have to overload the '__init__' and 'forward'
    you don't have to initialize rState(prepared in __call__) and dState(from scorer),
    but use 'self.rState' and self.dState directly

    you can also use 'self.inited_state' to gain the decoder init input state
    but actually 'self.dState' is the same as 'self.inited_state' at first
    '''
    def __init__(self,sos,eos,max_len=50):
        self.sos=sos
        self.eos=eos
        self.max_len=max_len
        self.logzero=-torch.finfo(torch.float16).min**2
    def init_dState(self,rState):
        if hasattr(self,"inited_state") and len(self.init_state)>=len(rState):
            return self.init_state[:len(rState)]
        
        feats=rState.feats
        self.batch_size=feats.size(0)

        self.inited_dState=State(
            torch.tensor([[self.sos]]*self.batch_size,
                         device=feats.device),
            torch.tensor([1]*self.batch_size,
                         device=feats.device)
        )

        return self.inited_dState
    
    def initState(self,rawState):
        '''
        init decoder state,
        include rawState and dState
        '''
        self.rState=rawState
        self.dState=self.init_dState(self.rState)
        self.device=self.rState.feats.device
    
    def __call__(self,
                 rawState: State,
                 decoders: Dict[str,EvalModule]):
        '''
        init encode state from the first scorer
        init decode state State([[sos]],[1])
        '''
        self.initState(rawState)

        '''
        begin scoring and search
        '''
        return self.forward({
            name:decoder.scorer(rawState,weight) \
                for name,(decoder,weight) in decoders.items()
        })
    
    def forward(self,scorers: Dict[str,Scorer]):
        '''
        x shape : (b, t, f)
        xl shape: (b, )
        '''
        raise NotImplementedError

class Beam(DecodeSearch):
    '''
    decoder of seq2seq
    '''
    def __init__(self,
                 sos,
                 eos,
                 vocab_size,
                 max_len=50,
                 beam_size=5,
                 blank=0,
                 ):
        super().__init__(sos,eos,max_len)
        self.vocab_size=vocab_size
        self.beam_size=beam_size
        self.exclude=("ctc",)
        self.blank=blank
    def topk(self,score:torch.Tensor,k:int):
        '''
        score shape: (beam_size, vocab_size)

        find the topk in score,
        detect the ends and select out

        return: 
        k_pref  :prefix keep ids            (num_beams, )
        e_pref  :prefix ends ids            (num_beams, )
        k_pred  :cur keep    ids            (num_beams, 1)
        e_pred  :cur ends    ids            (num_beams, 1)
        k_prob  :cur prefix prob (keep)     (num_beams, 1)
        e_prob  :cur prefix prob (ends)     (num_beams, 1)

        '''
        vocab_size=score.size(-1)
        prob,pred=score.reshape(-1).topk(k=k)
        pref=pred//vocab_size

        prob,pred=prob.view(-1,1),pred.view(-1,1)%vocab_size

        # temp_pred=self.toVocabIdx(pref,pred,l,r)
        # remains=(temp_pred!=self.eos).flatten()
        remains=(pred!=self.eos).flatten()

        keep=remains.nonzero(as_tuple=True)
        ends=(~remains).nonzero(as_tuple=True)


        return (pref[keep],pref[ends],
                pred[keep],pred[ends],
                prob[keep],prob[ends],
                )
        return keep,ends,pref,pred,prob # if requires keep and ends

    def forward(self, scorers: Dict[str,Scorer]) -> EndedPrefixes:
        assert len(self.rState)==1,"Please set batch_size=1 or use BatchBeam instead!"
        prefixes=EndedPrefixes([])
        self.num_beams=self.beam_size
        k_prob=0.

        for _ in range(self.max_len):
            # add prefix  prob
            weighted_score=k_prob # (num_beams, vocab_size)
            # add current prob

            for score in [scorer.weighted_scoring(self.dState)\
                          for name,scorer in scorers.items()\
                            if name not in self.exclude]:
                weighted_score=score + weighted_score

            (k_pref, e_pref, 
             k_pred, e_pred, 
             k_prob, e_prob)=self.topk(weighted_score,
                                       self.num_beams)

            #update num_beams
            self.num_beams=k_prob.size(0)

            #update prefixes
            e_full=self.dState[e_pref].concat_(State(e_pred)).feats
            prefixes.extend(Prefix(e_full,e_prob).unbatchify())

            #update States
            self.rState.setitem(k_pref)
            self.dState.setitem(k_pref).concat_(State(k_pred))

            if not self.num_beams:break
        else:
            pass
            # prefixes.extend(Prefix(self.dState.feats,k_prob).unbatchify())    

        # return prefixes.prefix
        return prefixes.sorted().prefix


class BatchBeam(Beam):
    '''
    Batch sample batch processing
    '''
    def batch_topk(self,score:torch.Tensor,batch_k):
        '''
        score shape: (batch_beam_size, vocab_size)

        find the topk in score,
        detect the ends and select out

        return: all below are lists
        k_pref  :prefix keep ids            [(num_beams, ) ]*batch_size
        e_pref  :prefix ends ids            [(num_beams, ) ]*batch_size
        k_pred  :cur keep    ids            [(num_beams, 1)]*batch_size
        e_pred  :cur ends    ids            [(num_beams, 1)]*batch_size
        k_prob  :cur prefix prob (keep)     [(num_beams, 1)]*batch_size
        e_prob  :cur prefix prob (ends)     [(num_beams, 1)]*batch_size

        BeWare that k_pref and e_pref are prefix idx after being concated!!
              used to select prefix from dState(which is also concated
        '''

        return zip(*[
            (k_pref+self.batch_bound[i],
             e_pref+self.batch_bound[i],
             k_pred,
             e_pred,
             k_prob,
             e_prob)

             if batch_k[i] else
             
             (torch.tensor([]).long().to(score.device),)*2 + \
             (torch.zeros(0,1).long().to(score.device),)*4

             for i,(l,r) in enumerate(zip(self.batch_bound[:-1],self.batch_bound[1:]))\
                for k_pref,e_pref,k_pred,e_pred,k_prob,e_prob in (self.topk(score[l:r],batch_k[i]),)
        ])
        
    def forward(self, scorers: Dict[str,Scorer]) -> EndedPrefixes:
        prefixes=[EndedPrefixes([]) for _ in range(self.batch_size)]

        self.batch_bound=[i for i in range(self.batch_size+1)]
        self.batch_num_beams=[self.beam_size for _ in range(self.batch_size)]
        

        bk_prob=0.
        for _ in range(self.max_len):
            weighted_score=bk_prob
            # (batch_beam_size(actually is sum(last_beam_size)), vocab_size)
            for score in [scorer.weighted_scoring(self.dState)\
                          for name,scorer in scorers.items()\
                            if name not in self.exclude]:
                weighted_score=score+weighted_score 

            # weighted_score[:,0]=-100000000   

            (bk_pref, be_pref,
             bk_pred, be_pred,
             bk_prob, be_prob)=self.batch_topk(weighted_score,
                                               self.batch_num_beams)
            
        
            #update batch_bound
            for i in range(self.batch_size):
                self.batch_bound[i+1]=self.batch_bound[i]+bk_prob[i].size(0)

            #update batch_num_beams
            self.batch_num_beams=[k_prob.size(0) for k_prob in bk_prob]


            for scorer in scorers.values():
                scorer.selectNext(torch.concat(bk_pref),
                                  torch.concat(bk_pred).flatten())

            #update prefixes
            be_full=[self.dState[e_pref].concat_(State(e_pred)).feats\
                     for e_pref,e_pred in zip(be_pref,be_pred)]
            for i in range(self.batch_size):
                prefixes[i].extend(Prefix(be_full[i],be_prob[i]).unbatchify())

            temp_pref=[self.dState[k_pref].feats\
                       for k_pref in bk_pref]

            #update States
            bk_pref=torch.concat(bk_pref)
            bk_pred=torch.concat(bk_pred)

            self.rState.setitem(bk_pref)
            self.dState.setitem(bk_pref).concat_(State(bk_pred))

            bk_prob=torch.concat(bk_prob)

            if not any(self.batch_num_beams):break
        else:
            for i in range(self.batch_size):
                if temp_pref[i].size(0)==0:continue
                prefixes[i].extend(Prefix(temp_pref[i],torch.tensor([0.]*temp_pref[i].size(0),
                                                                   device=self.dState.device)).unbatchify())


        
        return [prefixes_per_sample.sorted().prefix for prefixes_per_sample in prefixes]



if __name__=="__main__":
    
    a=torch.tensor([[2,3,4,5,6],
                    [1,2,3,4,5]])
    
    i=torch.tensor([[1,1,1,1,0],
                    [0,1,0,0,0]])
    i=i.flatten()
    print(i.flatten().nonzero(as_tuple=True))
    # print(a[torch.arange(2)[:,None],i])
    # print(torch.max(torch.randn(40),dim=-1))
    # pass
    a=State(torch.randn(4,0,5))
    # print(a[:,1:2].valen)
    print(not a)

