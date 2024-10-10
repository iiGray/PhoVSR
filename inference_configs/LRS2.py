
import os
from model import *
from model.lm import *
from datas import *

from template.e2e.evaluator import PhoE2EEvaluator
from template.e2e.lengthpenalty import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


mode="test"

test_set=LRS2PhonemeDataset(mode="test",convert_gray=True)

model=PhonemeVSR(enum_layers=12,
                 dnum_layers=6,
                 idims=256,
                 hdims=2048,
                 num_heads=4,
                 dec_char_list_len=len(test_set.vocab),
                 phoneme_list_len=len(test_set.phoneme_vocab),
                 g2p=test_set.g2p,
                 p2g=test_set.p2g,
                 dropout=0.1,
                 norm_type=PreNorm,
                 use_conv=True
                 )


lm=RoformerLM(num_layers=16,
              idims=512,
              hdims=2056,
              num_heads=8,
              char_list_len=len(test_set.vocab),
              dropout=0.1,
              norm_type=AddNorm
              )

PhoE2EEvaluator.load_model(lm,model_name="LM_LRS2",device="cuda:0",evaluating=True,load_best=True)


lengthpenalty=LengthPenalty(len(test_set.vocab))

