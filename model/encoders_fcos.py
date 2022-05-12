import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary


from model.blocks import LayerStack, PositionwiseFeedForward, ResidualConnection, clone
from model.multihead_attention import MultiHeadedAttention


class TriModalEncoder(nn.Module):

    def __init__(self, cfg, d_model_A, d_model_V, d_model_T, d_model, dout_p, d_ff_A, d_ff_V, d_ff_T):
        super(TriModalEncoder, self).__init__()
        self.H = cfg.H
        self.N = cfg.N

        self.encoder_avt = Encoder(d_model_A, d_model_V, d_model_T, d_model, dout_p, self.H, self.N, d_ff_A, d_ff_V, d_ff_T)

    def forward(self, x, masks):
        """
        :param x: x: (audio(B, S, Da), video(B, S, Dv), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'AV_mask':(B, 1, S), 'T_mask':(B, 1, S)}
        :return: avt:(B, S, D_fcos)
        """
        A, V, T = x
        Avt, Tav, Vat, Tva = self.encoder_avt((A, V, T), masks)

        return (Avt, Tav, Vat, Tva)


class Encoder(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model_T, d_model, dout_p, H, N, d_ff_A, d_ff_V, d_ff_T):
        super(Encoder, self).__init__()
        self.d_model_A = d_model_A
        self.d_model_V = d_model_V
        self.d_model_T = d_model_T
        self.d_model = d_model
        self.drop = dout_p
        self.H = H
        self.N = N

        layer_AV = BiModalEncoderOne(d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V)
        self.encoder_AV = LayerStack(layer_AV, N)

        layer_TAV = BiModalEncoderTwo(d_model_A, d_model_T, d_model, dout_p, H, d_ff_A, d_ff_T)
        self.encoder_TAV = LayerStack(layer_TAV, N)

        layer_TVA = BiModalEncoderThree(d_model_V, d_model_T, d_model, dout_p, H, d_ff_V, d_ff_T)
        self.encoder_TVA = LayerStack(layer_TVA, N)

    def forward(self, x, masks):
        A, V, T = x

        Av, Va = self.encoder_AV((A, V), (masks['A_mask'], masks['V_mask']))

        Avt, Tav = self.encoder_TAV((Av, T), (masks['A_mask'], masks['T_mask']))

        Vat, Tva = self.encoder_TVA((Va, T), (masks['V_mask'], masks['T_mask']))

        return Avt, Tav, Vat, Tva


class BiModalEncoderOne(nn.Module):

    def __init__(self, d_model_M1, d_model_M2, d_model, dout_p, H, d_ff_M1, d_ff_M2):
        super(BiModalEncoderOne, self).__init__()
        # 自注意力层
        self.self_att_M1 = MultiHeadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.self_att_M2 = MultiHeadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)
        # 双模态注意力层
        self.cross_att_M1 = MultiHeadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.cross_att_M2 = MultiHeadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)
        # 位置全连接网络层
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)
        # 残差网络层
        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)

    def forward(self, x, masks):
        '''
        Forward:
            x:(A,V)
                A='audio' (B, Sa, Da)、  V='rgb'&'flow' (B, Sv, Dv),
            masks:(A_mask,V_mask)
                A_mask (B, 1, Sa), V_mask (B, 1, Sv)
            Output:
                Av:(B, Sa, Da), Va:(B, Sv, Da)
        '''
        M1, M2 = x
        M1_mask, M2_mask = masks

        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_cr_att_M1(M1): return self.cross_att_M1(M1, M2, M2, M2_mask)
        def sublayer_cr_att_M2(M2): return self.cross_att_M2(M2, M1, M1, M1_mask)
        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        # 2. Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_cr_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_cr_att_M2)

        # 3. Feed-forward
        M1m2 = self.res_layers_M1[2](M1m2 , sublayer_ff_M1)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class BiModalEncoderTwo(nn.Module):

    def __init__(self, d_model_A, d_model_T, d_model, dout_p, H, d_ff_A, d_ff_T):
        super(BiModalEncoderTwo, self).__init__()
        # 自注意力层
        self.self_att_M1 = MultiHeadedAttention(d_model_A, d_model_A, d_model_A, H, dout_p, d_model)
        self.self_att_M2 = MultiHeadedAttention(d_model_T, d_model_T, d_model_T, H, dout_p, d_model)
        # 双模态注意力层
        # TAv, Avt, Vat, TVa --> M1,M2,M3,M4
        self.cross_att_M1 = MultiHeadedAttention(d_model_A, d_model_T, d_model_T, H, dout_p, d_model)
        self.cross_att_M2 = MultiHeadedAttention(d_model_T, d_model_A, d_model_A, H, dout_p, d_model)
        # 位置全连接网络层
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_A, d_ff_A, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_T, d_ff_T, dout_p)
        # 残差网络层
        self.res_layers_M1 = clone(ResidualConnection(d_model_A, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_T, dout_p), 3)

    def forward(self, x, masks):
        # M1,M2,M3,M4 --> T,A,V,T
        M1, M2 = x
        M1_mask, M2_mask = masks

        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_cr_att_M1(M1): return self.cross_att_M1(M1, M2, M2, M2_mask)
        def sublayer_cr_att_M2(M2): return self.cross_att_M2(M2, M1, M1, M1_mask)
        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        # 2. Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_cr_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_cr_att_M2)

        # 3. Feed-forward
        M1m2 = self.res_layers_M1[2](M1m2 , sublayer_ff_M1)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class BiModalEncoderThree(nn.Module):

    def __init__(self, d_model_V, d_model_T, d_model, dout_p, H, d_ff_V, d_ff_T):
        super(BiModalEncoderThree, self).__init__()
        # 自注意力层
        self.self_att_M1 = MultiHeadedAttention(d_model_V, d_model_V, d_model_V, H, dout_p, d_model)
        self.self_att_M2 = MultiHeadedAttention(d_model_T, d_model_T, d_model_T, H, dout_p, d_model)
        # 双模态注意力层
        # TAv, Avt, Vat, TVa --> M1,M2,M3,M4
        self.cross_att_M1 = MultiHeadedAttention(d_model_V, d_model_T, d_model_T, H, dout_p, d_model)
        self.cross_att_M2 = MultiHeadedAttention(d_model_T, d_model_V, d_model_V, H, dout_p, d_model)
        # 位置全连接网络层
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_V, d_ff_V, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_T, d_ff_T, dout_p)
        # 残差网络层
        self.res_layers_M1 = clone(ResidualConnection(d_model_V, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_T, dout_p), 3)

    def forward(self, x, masks):
        # M1,M2 --> V,T
        M1, M2 = x
        M1_mask, M2_mask = masks

        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_cr_att_M1(M1): return self.cross_att_M1(M1, M2, M2, M2_mask)
        def sublayer_cr_att_M2(M2): return self.cross_att_M2(M2, M1, M1, M1_mask)
        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        # 2. Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_cr_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_cr_att_M2)

        # 3. Feed-forward
        M1m2 = self.res_layers_M1[2](M1m2 , sublayer_ff_M1)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


if __name__ == '__main__':
    encoder_avt = TriModalEncoder(
                d_model_A=128,
                d_model_V=1024,
                d_model_T=300,
                d_model=1024,
                dout_p=0.1,
                H=4,
                N=2,
                d_ff_A=128*2,
                d_ff_V=1024*2,
                d_ff_T=300*2)

    a = torch.randn(2,24,128)
    v = torch.randn(2,10,1024)
    t = torch.randn(2,30,300)

    masks = {'A_mask': None, 'V_mask': None, 'T_mask': None}

    Avt, Tav, Vat, Tva = encoder_avt((a,v,t), masks)
    # Av, Va = encoder_avt((a,v,t), masks)

    print('a注意v注意t:', Avt.shape)
    print('t注意a注意v:', Tav.shape)
    print('v注意a注意t:', Vat.shape)
    print('t注意v注意a:', Tva.shape)

    # print(summary(encoder_avt, (a, v, t), masks))
