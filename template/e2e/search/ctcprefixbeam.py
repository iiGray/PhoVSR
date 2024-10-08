if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from template.e2e.search.beamsearch import *
import math

class CTCPrefixBeam(DecodeSearch):
    def __init__(self,
                 sos,
                 eos,
                 vocab_size,
                 ctc_beam_size=30,
                 blank=0):
        super().__init__(sos, eos)
        self.vocab_size=vocab_size
        self.ctc_beam_size=ctc_beam_size
        self.blank=blank

        self.vocab_idx=torch.arange(self.vocab_size)[None,:]

    def logsumexp(self,*args):
        if all(a == self.logzero for a in args):
            return self.logzero
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max) for a in args))
        return a_max + lsp


    def forward(self, scorers: Dict[str,Scorer]) -> EndedPrefixes:
        prefixes=EndedPrefixes([])
        X=scorers["ctc"].state.feats
        assert X.size(0)==1,"Please set batch_size=1"

        batch_size,num_steps,vocab_size=X.shape

        '''select main idx'''
        top_num_steps_idx=set()
        for i in range(num_steps):
           prob, ids=X[0,i].topk(num_steps)    
           for p,j in zip(prob.tolist(),ids.tolist()):
               top_num_steps_idx.add((p,j))
        
        selected=sorted(list(top_num_steps_idx),reverse=True)[:num_steps]
        idx=[s[1] for s in selected]
        X=X[:,:,idx]

        pref = [(tuple(), (0, self.logzero))]
        for t in range(num_steps):
            new_pref = defaultdict(lambda: (self.logzero,
                                            self.logzero))
            for p, (pb, pn) in pref:
                for i in range(num_steps):
                    prob = X[0, t, i].item()
                    if i == self.blank:
                        npb, npn = new_pref[p]
                        npb = self.logsumexp(npb, pb + prob, pn + prob)
                        new_pref[p] = (npb, npn)
                        continue

                    last = p[-1] if p else None

                    npb, npn = new_pref[p + (i,)]
                    
                    if i == last:
                        npn = self.logsumexp(npn,
                                             pb + prob)
                    else:
                        npn = self.logsumexp(npn,
                                             pb + prob,
                                             pn + prob)
                        
                    new_pref[p + (i,)] = (npb, npn)

                    if i == last:
                        npb, npn = new_pref[p]
                        npn = self.logsumexp(npn, pn + prob)
                        new_pref[p] = (npb, npn)
            # top beam_size
            pref = sorted(new_pref.items(),
                          key=lambda x: self.logsumexp(*x[1]),
                          reverse=True)[:self.ctc_beam_size]
            
        p=[idx[k] for k in pref[0][0]]
        ret=[p[i] for i in range(len(p)) \
             if p[i]!=self.blank and (i==0 or p[i]!=p[i-1])]
        prefixes.append(Prefix([ret],0.))
        return prefixes.prefix
    
if __name__=="__main__":
    pass