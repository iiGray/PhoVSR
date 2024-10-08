if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from template.head import *
from template.e2e.search import DecodeSearch,Beam,BatchBeam

from template.evaluator import Evaluator
from template.e2e.utils import Vocab,State,EvalModule
from template.e2e.indicator import Indicator,Bleu,CER

class E2EEvaluator(Evaluator):
    def __init__(self,
                 model,
                 model_name :str,
                 load_best:bool,
                 vocab      :Vocab,
                 search     :Union[Beam,BatchBeam]=BatchBeam,
                 decoders   :Dict[str,Tuple[EvalModule,float]]={
                     "decoder":(None,None),
                     },
                 indicators={
                    "bleu":Bleu(),
                    "cer" :CER()
                 },
                 device="cuda:0",
                 update_interval=300
                 ):
        assert hasattr(model,"encode"),"Model MUST be an E2E model and has method 'encode'!"
        super().__init__(model,model_name,load_best,device,update_interval)
    
        self.vocab=vocab
        self.search=search
        self.decoders=decoders
        self.indicators.update(indicators)
        self.sep=self.vocab[self.search.sep]

    def evaluate(self,
                 preds:Union[List,torch.Tensor],
                 ys   :torch.Tensor,
                 yls  :torch.Tensor,
                 *args
                 ):
        '''
        return  : tuple(int,int), error nums,total nums are the sum of all sample in one batch
        '''
        error,total=0,0

        for i in range(ys.size(0)):
            pred,y,yl=preds[i],ys[[i]],yls[[i]]
            if isinstance(pred,list):
                pred=torch.tensor(pred)
            pred=pred.view(-1)[1:-1] # exclude sos and eos
            y=y.view(-1)[:yl]
        
            pred,y=self.reprocess(pred.tolist()),self.reprocess(y.tolist())
            for name,indicator in self.indicators.items():
                indicator.add(pred,y)
    def reprocess(self,x):
        return x
        

    @Evaluator.no_grad
    def predict(self, x,xl):
        eState=State(*self.model.encode(x.to(self.device),xl.to(self.device)))

        out=self.search(
            rawState=eState,
            decoders=self.decoders
        )
        if x.size(0)==1 and type(self.search) is Beam:
            return out[:1]
        return [o[:1] for o in out]
    
    def to_sentence(self,x:list):
        if not x or isinstance(x[0],str):
            return "".join(x).replace(self.sep," ")
        return [self.to_sentence(k) for k in x]

    def concat_sentence(self,x:list):
        if isinstance(x,str):
            return x.replace(self.sep," ").lstrip(self.vocab[self.search.sos]
                            ).rstrip(" "+self.vocab[self.search.eos])
        if isinstance(x[0],str):return "\n".join(x)
        return [self.concat_sentence(k) for k in x]

    def print(self, pred, y, yl,*args):
        
        pred = [k for k in pred if k]
        if not pred:return

        if isinstance(pred[0][0],list):
            for i,unpaded_y in enumerate(rnn_utils.unpad_sequence(y.T,yl)):
                self.print(pred[i],unpaded_y[None,:],yl[[i]])
            return
        pred=self.vocab[pred[:1]]

        y=self.vocab[y.tolist()]
        
        ps=self.to_sentence(pred)

        ps=[self.concat_sentence(k) for k in ps]

        ys=self.to_sentence(y)

        print("pred:",("\n"+"--"*self.search.max_len+"\n").join(ps))
        print("ref :",("\n"+"--"*self.search.max_len+"\n").join(ys))



class PhoE2EEvaluator(E2EEvaluator):
    '''
    This class can calculate the WER of a language containing spaces, such as English
    '''
    def __init__(self,
                 model,
                 model_name :str,
                 load_best  :bool,
                 vocab      :Vocab,
                 search     :Union[Beam,BatchBeam]=BatchBeam,
                 decoders   :Dict[str,Tuple[EvalModule,float]]={
                     "decoder":(None,None),
                     },
                 indicators={
                    "bleu":Bleu(),
                    "cer" :CER()
                 },
                 device="cuda:0",
                 update_interval=300,
                 mode:Literal["cer","wer"]="wer"
                 ):
        super().__init__(model,model_name,load_best,vocab,search,decoders,indicators,device,update_interval)
        self.mode=mode

        self.sep="<sep>"
    
    def reprocess(self,x):
        x="".join(self.vocab[x])
        if self.mode=="cer":
            x=list(x.replace(self.sep," ").strip())
        else:
            x=x.split(self.sep)

        return x