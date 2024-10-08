if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from template.e2e.search.beamsearch import *

class BatchBeamWithCTCPrefix(BatchBeam):
    def __init__(self,
                 sos,
                 eos,
                 vocab_size,
                 max_len=50,
                 beam_size=5,
                 ctc_beam_size=30,
                 blank=0):
        super().__init__(sos, eos, vocab_size, max_len, beam_size)
        self.ctc_beam_size=ctc_beam_size
        self.blank=blank
        self.exclude=("ctc",)

    def forward(self, scorers: Dict[str,Scorer]) -> EndedPrefixes:
        prefixes=[EndedPrefixes([]) for _ in range(self.batch_size)]

        self.batch_bound=[i for i in range(self.batch_size+1)]
        self.batch_num_beams=[self.beam_size for _ in range(self.batch_size)]
        

        bk_prob=0.

        scorers["ctc"].preProcess(self.blank,self.sos,self.eos)

        for _ in range(self.max_len):
            # (batch_beam_size, vocab_size)
            weighted_score=bk_prob
            
            for scorer in (scorer for name,scorer in scorers.items()\
                            if name not in self.exclude):
                weighted_score=(scorer.weighted_scoring(self.dState) + weighted_score)
            # (batch_beam_size, beam_size* vocab_size)  ->  (batch_beam_size, ctc_beam_size)
            
            '''
            calculate ctc score
            '''
            # (batch_beam_size, ctc_beam_size)
            ctc_beam_idx=weighted_score.topk(k=self.ctc_beam_size,dim=-1)[1]


            weighted_score=scorers["ctc"].weighted_scoring(self.dState, ctc_beam_idx) \
                + weighted_score
            
            '''
            bk_pred and be_pred is not real,
            need to be converted from ctc_beam_idx into vocab idx
            '''
            (bk_pref, be_pref,
             bk_pred, be_pred,
             bk_prob, be_prob)=self.batch_topk(weighted_score,
                                               self.batch_num_beams)

            #update batch_bound
            for i in range(self.batch_size):
                self.batch_bound[i+1]=self.batch_bound[i]+bk_prob[i].size(0)

            #update batch_num_beams
            self.batch_num_beams=[k_prob.size(0) for k_prob in bk_prob]


            '''
            ctc scorer select Next State

            the non-public states' being updated ( self.rState is public state)            
            Note that current bk_pred is on ctcBeamIdx
            '''
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
            # print("Warning: Some prefix didn't end  !!")

            for i in range(self.batch_size):
                if temp_pref[i].size(0)==0:continue
                prefixes[i].extend(Prefix(temp_pref[i],torch.tensor([0.]*temp_pref[i].size(0),
                                                                   device=self.dState.device)).unbatchify())

        return [prefixes_per_sample.sorted().prefix for prefixes_per_sample in prefixes]

            


if __name__=="__main__":
    a=torch.zeros((4,10))
    b=torch.randn(4,3)
    c=torch.tensor([[0,2,5],
                    [3,4,5],
                    [6,7,8],
                    [0,1,9]])
    
    a[torch.arange(c.size(0))[:,None],c]=b
    print(a)