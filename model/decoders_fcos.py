import torch
import torch.nn as nn

from model.blocks import LayerStack, PositionwiseFeedForward, ResidualConnection, BridgeConnection
from model.multihead_attention import MultiHeadedAttention
from torchsummaryX import summary


class TriModalDecoderLayer(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model_T, d_model_C, d_model, dout_p, H, d_ff_C):
        super(TriModalDecoderLayer, self).__init__()

        # self attention
        self.res_layer_self_att = ResidualConnection(d_model_C, dout_p)
        self.self_att = MultiHeadedAttention(d_model_C, d_model_C, d_model_C, H, dout_p, d_model)
        # encoder attention
        self.res_layer_enc_att_Avt = ResidualConnection(d_model_C, dout_p)
        self.res_layer_enc_att_Tav = ResidualConnection(d_model_C, dout_p)
        self.res_layer_enc_att_Vat = ResidualConnection(d_model_C, dout_p)
        self.res_layer_enc_att_Tva = ResidualConnection(d_model_C, dout_p)
        self.enc_att_Avt = MultiHeadedAttention(d_model_C, d_model_A, d_model_A, H, dout_p, d_model)
        self.enc_att_Tav = MultiHeadedAttention(d_model_C, d_model_T, d_model_T, H, dout_p, d_model)
        self.enc_att_Vat = MultiHeadedAttention(d_model_C, d_model_V, d_model_V, H, dout_p, d_model)
        self.enc_att_Tva = MultiHeadedAttention(d_model_C, d_model_T, d_model_T, H, dout_p, d_model)
        # bridge
        self.bridge = BridgeConnection(4*d_model_C, d_model_C, dout_p)
        # feed forward residual
        self.res_layer_ff = ResidualConnection(d_model_C, dout_p)
        self.feed_forward = PositionwiseFeedForward(d_model_C, d_ff_C, dout_p)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: AVT: (B, Savt, d_model)
            masks (AVT_mask: (B, 1, Savt); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: AVT: (B, Savt, Davt)
        '''
        C, memory = x
        Avt, Tav, Vat, Tva = memory

        def sublayer_self_att(C): return self.self_att(C, C, C, masks['C_mask'])
        def sublayer_enc_att_Avt(C): return self.enc_att_Avt(C, Avt, Avt, masks['A_mask'])
        def sublayer_enc_att_Tav(C): return self.enc_att_Tav(C, Tav, Tav, masks['T_mask'])
        def sublayer_enc_att_Vat(C): return self.enc_att_Vat(C, Vat, Vat, masks['V_mask'])
        def sublayer_enc_att_Tva(C): return self.enc_att_Tva(C, Tva, Tva, masks['T_mask'])
        sublayer_feed_forward = self.feed_forward

        # 1. Self Attention
        # (B, Sc, Dc)
        C = self.res_layer_self_att(C, sublayer_self_att)

        # 2. Encoder-Decoder Attention
        # (B, Sc, Dc) each
        CAvt = self.res_layer_enc_att_Avt(C, sublayer_enc_att_Avt)
        CTav = self.res_layer_enc_att_Tav(C, sublayer_enc_att_Tav)
        CVat = self.res_layer_enc_att_Vat(C, sublayer_enc_att_Vat)
        CTva = self.res_layer_enc_att_Tva(C, sublayer_enc_att_Tva)
        # (B, Sc, 2*Dc)
        C = torch.cat([CAvt, CTav, CVat, CTva], dim=-1)
        # bridge: (B, Sc, Dc) <- (B, Sc, 2*Dc)
        C = self.bridge(C)

        # 3. Feed-Forward
        # (B, Sc, Dc) <- (B, Sc, Dc)
        C = self.res_layer_ff(C, sublayer_feed_forward)

        return C, memory


class TriModelDecoder(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model_T, d_model_C, d_model, dout_p, H, N, d_ff_C):
        super(TriModelDecoder, self).__init__()
        layer = TriModalDecoderLayer(d_model_A, d_model_V, d_model_T, d_model_C, d_model, dout_p, H, d_ff_C)
        self.decoder = LayerStack(layer, N)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        # x is (C, memory)
        C, memory = self.decoder(x, masks)

        return C


if __name__ == '__main__':
    decoder = TriModelDecoder(
                d_model_A=128,
                d_model_V=1024,
                d_model_T=300,
                d_model_C=300,
                d_model=1024,
                dout_p=0.1,
                H=4,
                N=2,
                d_ff_C=300,
                )
    # Avt, Tav, Vat, Tva
    Avt = torch.randn(2,24,128)
    Tav = torch.randn(2,30,300)
    Vat = torch.randn(2,10,1024)
    Tva = torch.randn(2,30,300)
    c = torch.randn(2,15,300)

    masks = {'A_mask': None, 'V_mask': None, 'T_mask': None, 'C_mask': None}

    c = decoder((c,(Avt, Tav, Vat, Tva)), masks)

    print('c注意avt:', c.shape)

    print(summary(decoder, (c,(Avt, Tav, Vat, Tva)), masks))
