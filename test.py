#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json,hydra
import torch.utils.data as tud




from template.e2e.search import BatchBeamWithCTCPrefix
from template.e2e.indicator import CER
from template.e2e.evaluator import PhoE2EEvaluator



@hydra.main(version_base=None, config_path="inference_configs", config_name="default")
def main(cfg):
    
    device=cfg.device
    model_name=cfg.model_name

    configs=json.load(open(f"./inference_configs/{model_name}.json","r"))

    if model_name=="CMLR":
        from inference_configs.CMLR import model,lm,test_set,lengthpenalty
    if model_name=="CMLR":
        '''We'll upload it soon.'''
        pass

    bcsearch=BatchBeamWithCTCPrefix(sos=test_set.vocab["<sos>"],
                                eos=test_set.vocab["<eos>"],
                                sep=test_set.vocab[" "],
                                vocab_size=len(test_set.vocab),
                                beam_size=cfg.beam_size,
                                max_len=configs["max_len"],
                                ctc_beam_size=1,# this argument is useless in PhonemeCTCDecoder,but meanful in CTCDecoder
                                blank=0
                                )
    w=configs["weights"]
    evaler=PhoE2EEvaluator(model,
                    model_name="CMLR",
                    vocab=test_set.vocab,
                    device=device,
                    load_best=True,
                    search=bcsearch,
                    indicators={
                        "cer" if model_name=="CMLR" else "wer":CER(),
                    },
                    decoders={
                        "decoder": (model.decoder,w["decoder"]),
                        "lm"     : (lm  ,w["lm"]),
                        
                        "lengthpenalty":(lengthpenalty,w["lengthpenalty"]),
                        "ctc"    : (model.ctc                , w["ctc"]),   
                    },
                        update_interval=120,
                    mode="cer" if model_name=="CMLR" else "wer"
                    )
    
    evaler(tud.DataLoader(test_set,
                          batch_size=cfg.batch_size,
                          collate_fn=test_set.collate),mode=cfg.mode)


if __name__ == '__main__':
    main()
