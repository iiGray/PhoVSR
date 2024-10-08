import sys
sys.path.append(".")

from model import *
from datas import *
from template.e2e.search import BatchBeamWithCTCPrefix

from template.e2e.evaluator import E2EEvaluator,PhoE2EEvaluator

import os
from typing import List,Tuple
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from template.e2e.indicator import CER
from model.lm import *

test_set=CMLRPhonemeDataset(mode="test",convert_gray=True,augment=False)


from template.e2e.lengthpenalty import *


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
            use_conv=True,
            )


lm=RoformerLM(num_layers=16,
                 idims=512,
                 hdims=2056,
                 num_heads=8,
                 char_list_len=len(test_set.vocab),
                 dropout=0.1,
                 norm_type=AddNorm,
                 )
PhoE2EEvaluator.load_model(lm,model_name="LM_CMLR",device="cuda:0",evaluating=True,load_best=True)

bcsearch=BatchBeamWithCTCPrefix(sos=test_set.vocab["<sos>"],
                                eos=test_set.vocab["<eos>"],
                                sep=test_set.vocab["<sep>"],
                                vocab_size=len(test_set.vocab),
                                beam_size=20,
                                max_len=40,
                                ctc_beam_size=30,
                                blank=0
                                )
evaler1=E2EEvaluator(model,
                 model_name="CMLR",
                 vocab=test_set.vocab,
                 device="cuda:0",
                load_best=True,
                 search=bcsearch,
                indicators={
                    "cer":CER(),
                },
                 decoders={
                     "decoder": (model.decoder, 0.9),
                     "lm"     : (lm  ,0.6),
                     
                     "lengthPenalty":(LengthPenalty(len(test_set.vocab)),0.5),
                     "ctc"    : (model.ctc                , 0.1),   
                 },
                     update_interval=120,
                 )


evaler1(tud.DataLoader(test_set,
                       batch_size=2,
                       collate_fn=test_set.collate
                       ),
                       mode="show")
