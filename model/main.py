if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.frontend import *
from model.backend import *
from model.loss import *
class VSR(nn.Module):
    def __init__(self,
                 enum_layers,
                 dnum_layers,
                 idims,
                 hdims,
                 num_heads,
                 enc_char_list_len=None,
                 dec_char_list_len=None,
                 dropout=0,
                 use_conv=True,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.frontEnd=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
        
        self.encoder=RelEncoder(
            enum_layers,
            idims,
            hdims,
            num_heads,
            enc_char_list_len,
            dropout,
            norm_type,
            use_conv=use_conv
        )
        self.decoder=Decoder(
            dnum_layers,
            idims,
            hdims,
            num_heads,
            dec_char_list_len,
            dropout,
            norm_type
        )
        
        self.ctc=CTCdecoder(idims,
                            dec_char_list_len)


    def encode(self,x,x_valid_len):
        x=self.frontEnd(x,x_valid_len)
        
        return self.encoder(x,x_valid_len)
    def decode(self,y,y_valid_len,enc_out):
        return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
    def forward(self,x,x_valid_len,y,y_valid_len):
        enc_state=self.encode(x,x_valid_len)

        dec_attention,ctc_out=self.decode(y,y_valid_len,enc_state)

        return dec_attention,ctc_out



class PhonemeVSR(nn.Module):
    def __init__(self,
                 enum_layers,
                 dnum_layers,
                 idims,
                 hdims,
                 num_heads,
                 enc_char_list_len=None,
                 dec_char_list_len=None,
                 phoneme_list_len=None,
                 w2p=None,
                 p2w=None,
                 _c2p=None,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True,
                 ):
        super().__init__()
        self.frontEnd=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
        self.encoder=RelEncoder(
            enum_layers,
            idims,
            hdims,
            num_heads,
            enc_char_list_len,
            dropout,
            norm_type,
            use_conv=use_conv
        )
        self.decoder=Decoder(
            dnum_layers,
            idims,
            hdims,
            num_heads,
            dec_char_list_len,
            dropout,
            norm_type
        )
        
        self.ctc=PhonemeCTCdecoder(idims,
                                   phoneme_list_len,
                                   w2p,
                                   p2w,
                                   _c2p)


    def encode(self,x,x_valid_len):
        x=self.frontEnd(x,x_valid_len)
#         return x,x_valid_len
        return self.encoder(x,x_valid_len)
    def decode(self,y,y_valid_len,enc_out):
        
        
        return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
    def forward(self,x,x_valid_len,y,y_valid_len):
        enc_state=self.encode(x,x_valid_len)

        dec_attention,ctc_out=self.decode(y,y_valid_len,enc_state)

        return dec_attention,ctc_out
    

class PhonemeVSRAux(nn.Module):
    def __init__(self,
                 enum_layers,
                 dnum_layers,
                 idims,
                 hdims,
                 num_heads,
                 enc_char_list_len=None,
                 dec_char_list_len=None,
                 phoneme_list_len=None,
                 w2p=None,
                 p2w=None,
                 _c2p=None,
                 dropout=0,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm,
                 use_conv=True,
                 ):
        super().__init__()
        self.encoder=RelEncoderAux(
            enum_layers,
            idims,
            hdims,
            num_heads,
            enc_char_list_len,
            dropout,
            norm_type,
            use_conv=use_conv
        )
        
        
        
        
        
        self.decoder=Decoder(
            dnum_layers,
            idims,
            hdims,
            num_heads,
            dec_char_list_len,
            dropout,
            norm_type
        )
        
        self.ctc=PhonemeCTCdecoder(idims,
                                   phoneme_list_len,
                                   w2p,
                                   p2w,
                                   _c2p)


    def encode(self,x,x_valid_len):
        return self.encoder(x,x_valid_len)[:2]
    
    def decode(self,y,y_valid_len,enc_out):
        
        
        return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
    def forward(self,x,x_valid_len,y,y_valid_len):
        enc_f,enc_valid_len,L1Loss=self.encoder(x,x_valid_len)

        dec_attention,ctc_out=self.decode(y,y_valid_len,(enc_f,enc_valid_len))

        return dec_attention,ctc_out,L1Loss

    
# class PhonemeRoVSR(nn.Module):
#     def __init__(self,
#                  enum_layers,
#                  dnum_layers,
#                  idims,
#                  hdims,
#                  num_heads,
#                  enc_char_list_len=None,
#                  dec_char_list_len=None,
#                  phoneme_list_len=None,
#                  w2p=None,
#                  p2w=None,
#                  _c2p=None,
#                  dropout=0,
#                  norm_type:Union[PreNorm,AddNorm]=AddNorm,
#                  use_conv=True,
#                  ):
#         super().__init__()
#         self.frontEnd=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
#         self.encoder=RoEncoder(
#             enum_layers,
#             idims,
#             hdims,
#             num_heads,
#             enc_char_list_len,
#             dropout,
#             norm_type,
#             use_conv=use_conv
#         )
#         self.decoder=RoDecoder(
#             dnum_layers,
#             idims,
#             hdims,
#             num_heads,
#             dec_char_list_len,
#             dropout,
#             norm_type
#         )
        
#         self.ctc=PhonemeCTCdecoder(idims,
#                                    phoneme_list_len,
#                                    w2p,
#                                    p2w,
#                                    _c2p)


#     def encode(self,x,x_valid_len):
#         x=self.frontEnd(x,x_valid_len)
        
#         return self.encoder(x,x_valid_len)
#     def decode(self,y,y_valid_len,enc_out):
        
        
#         return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
#     def forward(self,x,x_valid_len,y,y_valid_len):
#         enc_state=self.encode(x,x_valid_len)

#         dec_attention,ctc_out=self.decode(y,y_valid_len,enc_state)

#         return dec_attention,ctc_out
   


# class PhonemeRVSR(nn.Module):
#     def __init__(self,
#                  enum_layers,
#                  dnum_layers,
#                  idims,
#                  hdims,
#                  num_heads,
#                  enc_char_list_len=None,
#                  dec_char_list_len=None,
#                  phoneme_list_len=None,
#                  w2p=None,
#                  p2w=None,
#                  _c2p=None,
#                  dropout=0,
#                  norm_type:Union[PreNorm,AddNorm]=AddNorm,
#                  use_conv=True,
#                  ):
#         super().__init__()
#         self.frontEnd=FrontEndModule(idims,grayMode=True,resnet_type="resnet18")
#         self.encoder=RelEncoder(
#             enum_layers,
#             idims,
#             hdims,
#             num_heads,
#             enc_char_list_len,
#             dropout,
#             norm_type,
#             use_conv=use_conv
#         )
#         self.decoder=RoDecoder(
#             dnum_layers,
#             idims,
#             hdims,
#             num_heads,
#             dec_char_list_len,
#             dropout,
#             norm_type
#         )
        
#         self.ctc=PhonemeCTCdecoder(idims,
#                                    phoneme_list_len,
#                                    w2p,
#                                    p2w,
#                                    _c2p)


#     def encode(self,x,x_valid_len):
#         x=self.frontEnd(x,x_valid_len)
        
#         return self.encoder(x,x_valid_len)
#     def decode(self,y,y_valid_len,enc_out):
        
        
#         return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
#     def forward(self,x,x_valid_len,y,y_valid_len):
#         enc_state=self.encode(x,x_valid_len)

#         dec_attention,ctc_out=self.decode(y,y_valid_len,enc_state)

#         return dec_attention,ctc_out
if __name__=="__main__":
    # print(globals()["net"])
    # print(locals())
    # print(globals())
    # a()
    import multiprocessing
    print(multiprocessing.cpu_count())
    pass