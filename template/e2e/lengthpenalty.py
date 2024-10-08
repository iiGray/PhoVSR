from torch._tensor import Tensor
from template.e2e.utils import *

class LengthPenalty(EvalModule):
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size

    def scoring(self, eState: State, dState: State, *args, **kwargs) -> Tensor:
        batch_beam_size=len(dState)
        return torch.ones((batch_beam_size,self.vocab_size),device=dState.device)
