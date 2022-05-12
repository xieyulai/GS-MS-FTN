import torch
import torch.nn as nn

from model.blocks import PositionalEncoder, VocabularyEmbedder
from model.decoders_fcos import TriModelDecoder
from model.encoders_fcos import TriModalEncoder
from model.generators_fcos import GeneratorFCOS


class TriModalTransformer(nn.Module):

    def __init__(self, cfg, train_dataset):
        super(TriModalTransformer, self).__init__()
        self.cfg = cfg
        self.pad_idx = train_dataset.pad_idx

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)

        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)  # (32,*,128)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)  # (32,*,1024)
        self.pos_enc_T = PositionalEncoder(cfg.d_model_text, cfg.dout_p)  # (32,*,300)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)  # (32,*,300)

        self.encoder = TriModalEncoder(cfg, cfg.d_model_audio, cfg.d_model_video, cfg.d_model_text, cfg.d_model,
                                       cfg.dout_p, cfg.d_ff_audio, cfg.d_ff_video, cfg.d_ff_text)

        self.decoder = TriModelDecoder(cfg.d_model_audio, cfg.d_model_video, cfg.d_model_text, cfg.d_model_caps,
                                       cfg.d_model, cfg.dout_p, cfg.H, cfg.N, cfg.d_ff_cap)

        self.generator = GeneratorFCOS(cfg.d_model_caps, train_dataset.trg_voc_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

    def forward(self, src, trg, masks):

        A = src['audio']
        V = src['rgb'] + src['flow']
        T = src['text']
        C = trg

        # (B, S, D) <- (B, S)
        C = self.emb_C(C)

        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        T = self.pos_enc_T(T)
        C = self.pos_enc_C(C)

        # notation: M1m2m2 (B, Sm1, Dm1), M1 is the target modality, m2 is the source modality
        AVT = self.encoder((A, V, T), masks)

        # (B, Sc, Dc)
        C_glove = self.decoder((C, AVT), masks)

        # (B, Sc, Vocabc) <- (B, Sc, Dc)
        C = self.generator(C_glove)

        return C

