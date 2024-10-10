if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.frontend import *
from model.backend import *

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
                 g2p=None,
                 p2g=None,
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
                                   g2p,
                                   p2g)


    def encode(self,x,x_valid_len):
        x=self.frontEnd(x,x_valid_len)
        return self.encoder(x,x_valid_len)
    
    def decode(self,y,y_valid_len,enc_out):    
        return self.decoder(y,y_valid_len,*enc_out),self.ctc(*enc_out) 
    
    def forward(self,x,x_valid_len,y,y_valid_len):
        enc_state=self.encode(x,x_valid_len)

        dec_attention,ctc_out=self.decode(y,y_valid_len,enc_state)

        return dec_attention,ctc_out
    

