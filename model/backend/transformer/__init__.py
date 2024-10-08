if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())

from model.backend.transformer.encoder import *
from model.backend.transformer.decoder import *

class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 idims,
                 hdims,
                 num_heads,
                 enc_char_list_len=None,
                 dec_char_list_len=None,
                 dropout=0,
                 use_cache=False,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.use_cache=use_cache

        self.encoder=Encoder(
            num_layers,
            idims,
            hdims,
            num_heads,
            enc_char_list_len,
            dropout,
            use_cache,
            norm_type
        )
        self.decoder=Decoder(
            num_layers,
            idims,
            hdims,
            num_heads,
            dec_char_list_len,
            dropout,
            norm_type
        )
    def encode(self,x,x_valid_len):
        return self.encoder(x,x_valid_len)
    def decode(self,y,y_valid_len,enc_out):
        return self.decoder(y,y_valid_len,*enc_out)

    def forward(self,x,x_valid_len,y,y_valid_len):
        return self.decode(
            y,
            y_valid_len,
            self.encode(x,x_valid_len)
        )

class RelTransformer(nn.Module):
    def __init__(self,
                 enum_layers,
                 dnum_layers,
                 idims,
                 hdims,
                 num_heads,
                 enc_char_list_len=None,
                 dec_char_list_len=None,
                 dropout=0,
                 use_cache=False,
                 norm_type:Union[PreNorm,AddNorm]=AddNorm
                 ):
        super().__init__()
        self.use_cache=use_cache

        self.encoder=RelEncoder(
            enum_layers,
            idims,
            hdims,
            num_heads,
            enc_char_list_len,
            dropout,
            use_cache,
            norm_type
        )
        # self.decoder=RelDecoder(
        #     num_layers,
        #     idims,
        #     hdims,
        #     num_heads,
        #     char_list_len,
        #     dropout,
        #     use_cache,
        #     norm_type
        # )
        self.decoder=Decoder(
            dnum_layers,
            idims,
            hdims,
            num_heads,
            dec_char_list_len,
            dropout,
            norm_type
        )
    def encode(self,x,x_valid_len):
        return self.encoder(x,x_valid_len)
    def decode(self,y,y_valid_len,enc_out):
        return self.decoder(y,y_valid_len,*enc_out)

    def forward(self,x,x_valid_len,y,y_valid_len):
        return self.decode(
            y,
            y_valid_len,
            self.encode(x,x_valid_len)
        )

def valid_normTransformer():
    t=RelTransformer(4,30,60,3,norm_type=PreNorm)
    x=torch.randn(3,8,30)
    y=torch.randn(3,5,30)

    x_valid_len=torch.tensor([4,6,7]).long()
    y_valid_len=torch.tensor([2,2,5]).long()

    out=t(x,y,x_valid_len,y_valid_len)
    print(out.shape)
    print(t)

if __name__=="__main__":
    valid_normTransformer()